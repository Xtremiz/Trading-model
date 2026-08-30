import time
import threading
from collections import deque

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# IMPORT YOUR PIPELINE
# ==========================================

from pipeline import pipeline


# ==========================================
# IMPORT TRADING FUNCTIONS
# ==========================================

from tradingfunc import (
    buy,
    sell,
    exit_buy_trade,
    exit_sell_trade
)


# ==========================================
# CONFIG
# ==========================================

SYMBOL = "GOLD"

TIMEFRAME = mt5.TIMEFRAME_M15

BUFFER_SIZE = 60

CHECK_INTERVAL = 1

MODEL_PATH = r"F:\Git-Hub\Trading model\models\best_gold_ann_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# TRADE EXECUTION
# ==========================================

# False = prediction only
# True = real trade execution
EXECUTE_TRADE = False


# ==========================================
# PROBABILITY THRESHOLD
# ==========================================

# Signal tabhi accept hoga jab
# kisi class ki probability 70% se ZYADA ho.

MIN_PROBABILITY = 0.70


# ==========================================
# MODEL ARCHITECTURE
# MUST MATCH TRAINING MODEL EXACTLY
# ==========================================

class MyNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.model = nn.Sequential(

            # 87 -> 32
            nn.Linear(
                num_features,
                32
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.Dropout(0.3),


            # 32 -> 256
            nn.Linear(
                32,
                256
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),


            # 256 -> 512
            nn.Linear(
                256,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.3),


            # 512 -> 64
            nn.Linear(
                512,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.3),


            # 64 -> 128
            nn.Linear(
                64,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),


            # 128 -> 3
            nn.Linear(
                128,
                3
            )
        )


    def forward(self, x):

        return self.model(x)


# ==========================================
# DEVICE
# ==========================================

print(
    "Using device:",
    DEVICE
)


# ==========================================
# LOAD CHECKPOINT
# ==========================================

try:

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

        weights_only=False

    )

except Exception as e:

    print(
        "Model loading failed:"
    )

    print(e)

    raise SystemExit


# ==========================================
# LOAD MODEL METADATA
# ==========================================

input_size = checkpoint["input_size"]

feature_names = checkpoint["feature_names"]

class_mapping = checkpoint["class_mapping"]


print("\nMODEL INFORMATION")

print(
    "Input size:",
    input_size
)

print(
    "Number of features:",
    len(feature_names)
)

print(
    "Class mapping:",
    class_mapping
)

print(
    "Minimum probability:",
    f"{MIN_PROBABILITY * 100:.0f}%"
)


# ==========================================
# CREATE MODEL
# ==========================================

model = MyNN(
    num_features=input_size
)


# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(
    DEVICE
)

model.eval()


print(
    "\nModel loaded successfully."
)


# ==========================================
# MT5 INITIALIZATION
# ==========================================

if not mt5.initialize():

    print(
        "MT5 initialization failed"
    )

    print(
        mt5.last_error()
    )

    raise SystemExit


print(
    "MT5 connected."
)


# ==========================================
# CHECK SYMBOL
# ==========================================

symbol_info = mt5.symbol_info(
    SYMBOL
)


if symbol_info is None:

    print(
        f"Symbol {SYMBOL} not found."
    )

    mt5.shutdown()

    raise SystemExit


if not symbol_info.visible:

    if not mt5.symbol_select(
        SYMBOL,
        True
    ):

        print(
            f"Could not select {SYMBOL}"
        )

        mt5.shutdown()

        raise SystemExit


# ==========================================
# GET COMPLETED CANDLES
# ==========================================

def get_completed_candles(count):

    rates = mt5.copy_rates_from_pos(

        SYMBOL,

        TIMEFRAME,

        # 0 = current forming candle
        # 1 = latest completed candle
        1,

        count

    )


    if rates is None:

        print(
            "Failed to get candles:"
        )

        print(
            mt5.last_error()
        )

        return None


    df = pd.DataFrame(
        rates
    )


    if df.empty:

        return None


    df["time"] = pd.to_datetime(

        df["time"],

        unit="s"

    )


    # Chronological order

    df = df.sort_values(

        "time"

    ).reset_index(

        drop=True

    )


    return df


# ==========================================
# INITIAL BUFFER
# ==========================================

initial_df = get_completed_candles(
    BUFFER_SIZE
)


if (

    initial_df is None

    or len(initial_df) < BUFFER_SIZE

):

    print(
        "Not enough completed candles."
    )

    mt5.shutdown()

    raise SystemExit


# ==========================================
# CREATE BUFFER
# ==========================================

buffer = deque(

    initial_df.to_dict(
        "records"
    ),

    maxlen=BUFFER_SIZE

)


print(

    f"Buffer initialized with "
    f"{len(buffer)} candles."

)


# ==========================================
# LAST PROCESSED CANDLE
# ==========================================

last_candle_time = buffer[-1]["time"]


print(
    "Last candle:",
    last_candle_time
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict():

    # --------------------------------------
    # BUFFER CHECK
    # --------------------------------------

    if len(buffer) < BUFFER_SIZE:

        print(
            "Not enough candles in buffer."
        )

        return None


    # --------------------------------------
    # BUFFER -> DATAFRAME
    # --------------------------------------

    df = pd.DataFrame(
        list(buffer)
    )


    df = df.sort_values(

        "time"

    ).reset_index(

        drop=True

    )


    # --------------------------------------
    # APPLY PIPELINE
    # --------------------------------------

    try:

        processed_df = pipeline(
            df.copy()
        )

    except Exception as e:

        print(
            "Pipeline error:"
        )

        print(e)

        return None


    if processed_df is None:

        print(
            "Pipeline returned None."
        )

        return None


    # --------------------------------------
    # ENSURE DATAFRAME
    # --------------------------------------

    if not isinstance(

        processed_df,

        pd.DataFrame

    ):

        print(
            "Pipeline must return a pandas DataFrame."
        )

        return None


    # --------------------------------------
    # CHECK REQUIRED FEATURES
    # --------------------------------------

    missing_features = [

        feature

        for feature in feature_names

        if feature not in processed_df.columns

    ]


    if missing_features:

        print(
            "\nMISSING FEATURES:"
        )

        print(
            missing_features
        )

        return None


    # --------------------------------------
    # REMOVE NaN ROWS
    # --------------------------------------

    processed_df = processed_df.dropna()


    if processed_df.empty:

        print(
            "No valid rows after pipeline."
        )

        return None


    # --------------------------------------
    # TAKE ONLY LATEST ROW
    # --------------------------------------

    latest_row = processed_df[
        feature_names
    ].iloc[[-1]]


    # --------------------------------------
    # DATAFRAME -> NUMPY
    # --------------------------------------

    X = latest_row.to_numpy(

        dtype=np.float32

    )


    # --------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------

    if not np.isfinite(
        X
    ).all():

        print(
            "NaN or Inf detected."
        )

        return None


    # --------------------------------------
    # INPUT SHAPE CHECK
    # --------------------------------------

    if X.shape[1] != input_size:

        print(
            "INPUT SIZE MISMATCH"
        )

        print(
            "Expected:",
            input_size
        )

        print(
            "Received:",
            X.shape[1]
        )

        return None


    # --------------------------------------
    # NUMPY -> TENSOR
    # --------------------------------------

    X_tensor = torch.tensor(

        X,

        dtype=torch.float32,

        device=DEVICE

    )


    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    with torch.no_grad():

        output = model(
            X_tensor
        )


        probabilities = torch.softmax(

            output,

            dim=1

        )


        confidence, prediction = torch.max(

            probabilities,

            dim=1

        )


    # --------------------------------------
    # CONVERT TO PYTHON
    # --------------------------------------

    prediction = prediction.item()

    confidence = confidence.item()


    # --------------------------------------
    # GET ALL PROBABILITIES
    # --------------------------------------

    probability_array = (

        probabilities[0]

        .cpu()

        .numpy()

    )


    hold_probability = float(
        probability_array[0]
    )

    buy_probability = float(
        probability_array[1]
    )

    sell_probability = float(
        probability_array[2]
    )


    # ======================================
    # PRINT ALL PROBABILITIES
    # ======================================

    print(
        "\n-----------------------------"
    )

    print(
        "MODEL PROBABILITIES"
    )

    print(
        "-----------------------------"
    )

    print(
        f"HOLD : {hold_probability * 100:.2f}%"
    )

    print(
        f"BUY  : {buy_probability * 100:.2f}%"
    )

    print(
        f"SELL : {sell_probability * 100:.2f}%"
    )


    # ======================================
    # FIND HIGHEST PROBABILITY
    # ======================================

    max_probability = max(

        hold_probability,

        buy_probability,

        sell_probability

    )


    # ======================================
    # 70% FILTER
    # ======================================

    if max_probability > MIN_PROBABILITY:

        # ----------------------------------
        # BUY
        # ----------------------------------

        if buy_probability == max_probability:

            signal = "BUY"

            reason = (

                f"BUY probability is "

                f"{buy_probability * 100:.2f}% "

                f"(> 70%)"

            )


        # ----------------------------------
        # SELL
        # ----------------------------------

        elif sell_probability == max_probability:

            signal = "SELL"

            reason = (

                f"SELL probability is "

                f"{sell_probability * 100:.2f}% "

                f"(> 70%)"

            )


        # ----------------------------------
        # HOLD
        # ----------------------------------

        else:

            signal = "HOLD"

            reason = (

                f"HOLD probability is "

                f"{hold_probability * 100:.2f}% "

                f"(> 70%)"

            )


    # ======================================
    # NOTHING ABOVE 70%
    # ======================================

    else:

        signal = "HOLD"

        reason = (

            "No class probability is above 70%. "

            f"Highest probability is "

            f"{max_probability * 100:.2f}%"

        )


    # ======================================
    # PRINT DECISION
    # ======================================

    print(
        "\n========= SIGNAL DECISION ========="
    )

    print(
        f"Signal: {signal}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        f"Reason: {reason}"
    )

    print(
        "==================================="
    )


    # ======================================
    # RETURN RESULT
    # ======================================

    return {

        "signal": signal,

        "prediction": prediction,

        "confidence": confidence,

        "probabilities": probability_array,

        "features": latest_row,

        "reason": reason

    }


# ==========================================
# GET CURRENT POSITION
# ==========================================

def get_current_position():

    positions = mt5.positions_get(
        symbol=SYMBOL
    )


    if not positions:

        return None


    position = positions[0]


    if position.type == mt5.POSITION_TYPE_BUY:

        return "BUY"


    elif position.type == mt5.POSITION_TYPE_SELL:

        return "SELL"


    return None


# ==========================================
# GET CURRENT FORMING CANDLE
# ==========================================

def get_current_candle():

    rates = mt5.copy_rates_from_pos(

        SYMBOL,

        TIMEFRAME,

        0,

        1

    )


    if rates is None or len(rates) == 0:

        return None


    candle = rates[0]


    return {

        "time": pd.to_datetime(

            candle["time"],

            unit="s"

        ),

        "open": candle["open"],

        "high": candle["high"],

        "low": candle["low"],

        "close": candle["close"],

        "tick_volume": candle["tick_volume"]

    }


# ==========================================
# LIVE STATUS
# ==========================================

def live_status():

    while True:

        candle = get_current_candle()

        tick = mt5.symbol_info_tick(
            SYMBOL
        )

        position = get_current_position()


        if candle and tick:

            candle_start = candle["time"]

            candle_end = (

                candle_start

                + pd.Timedelta(
                    minutes=15
                )

            )


            # --------------------------------------
            # BROKER SERVER TIME
            # --------------------------------------

            now = pd.to_datetime(

                tick.time,

                unit="s"

            )


            remaining = (

                candle_end - now

            )


            remaining_seconds = max(

                0,

                int(
                    remaining.total_seconds()
                )

            )


            minutes = (
                remaining_seconds // 60
            )

            seconds = (
                remaining_seconds % 60
            )


            # --------------------------------------
            # TRADE INFORMATION
            # --------------------------------------

            if position:

                positions = mt5.positions_get(
                    symbol=SYMBOL
                )


                if positions:

                    profit = positions[0].profit

                    trade_info = (

                        f"Trade: {position} | "

                        f"P/L: ${profit:.2f}"

                    )

                else:

                    trade_info = "No open trade"


            else:

                trade_info = "No open trade"


            # --------------------------------------
            # LIVE DISPLAY
            # --------------------------------------

            print(

                f"\r"

                f"Candle: "
                f"{candle_start.strftime('%H:%M')} | "

                f"O: {candle['open']:.2f} | "

                f"H: {candle['high']:.2f} | "

                f"L: {candle['low']:.2f} | "

                f"C: {candle['close']:.2f} | "

                f"Vol: {candle['tick_volume']} | "

                f"Next prediction: "
                f"{minutes:02}:{seconds:02} | "

                f"{trade_info} | "

                f"Commands: buy/sell/exit",

                end="",

                flush=True

            )


        time.sleep(1)


# ==========================================
# MANUAL TRADE COMMANDS
# ==========================================

def manual_trading():

    while True:

        command = input(

            "\n\nCommand "
            "(buy / sell / exit): "

        ).lower().strip()


        position = get_current_position()


        # ======================================
        # BUY
        # ======================================

        if command == "buy":

            if position is not None:

                print(

                    f"Already in {position}. "
                    f"Exit first."

                )

            else:

                result = buy()

                print(
                    "\nBUY RESULT:"
                )

                print(result)


        # ======================================
        # SELL
        # ======================================

        elif command == "sell":

            if position is not None:

                print(

                    f"Already in {position}. "
                    f"Exit first."

                )

            else:

                result = sell()

                print(
                    "\nSELL RESULT:"
                )

                print(result)


        # ======================================
        # EXIT
        # ======================================

        elif command == "exit":

            if position == "BUY":

                print(
                    "\nClosing BUY..."
                )

                exit_buy_trade()


            elif position == "SELL":

                print(
                    "\nClosing SELL..."
                )

                exit_sell_trade()


            else:

                print(
                    "\nNo open trade."
                )


        else:

            print(
                "\nInvalid command."
            )


# ==========================================
# START LIVE STATUS THREAD
# ==========================================

threading.Thread(

    target=live_status,

    daemon=True

).start()


# ==========================================
# START MANUAL COMMAND THREAD
# ==========================================

threading.Thread(

    target=manual_trading,

    daemon=True

).start()


# ==========================================
# START LIVE PREDICTION
# ==========================================

print(
    "\nLIVE PREDICTION STARTED\n"
)


try:

    while True:

        latest = get_completed_candles(1)


        if latest is not None and not latest.empty:

            latest_row = (
                latest.iloc[0].to_dict()
            )

            latest_time = (
                latest_row["time"]
            )


            # ==================================
            # NEW COMPLETED CANDLE
            # ==================================

            if latest_time != last_candle_time:

                print(
                    "\n\n================================"
                )

                print(
                    "NEW COMPLETED CANDLE"
                )

                print(
                    "Time:",
                    latest_time
                )

                print(
                    "================================"
                )


                # ----------------------------------
                # ADD CANDLE TO BUFFER
                # ----------------------------------

                buffer.append(
                    latest_row
                )


                last_candle_time = (
                    latest_time
                )


                # ==================================
                # PREDICTION
                # ==================================

                result = predict()


                if result is None:

                    print(
                        "Prediction skipped."
                    )


                else:

                    signal = (
                        result["signal"]
                    )

                    confidence = (
                        result["confidence"]
                    )

                    probabilities = (
                        result["probabilities"]
                    )

                    reason = (
                        result["reason"]
                    )


                    print(
                        "\n========= PREDICTION ========="
                    )


                    print(
                        f"Signal: {signal}"
                    )


                    print(
                        f"Confidence: "
                        f"{confidence * 100:.2f}%"
                    )


                    print(
                        f"HOLD: "
                        f"{probabilities[0] * 100:.2f}%"
                    )


                    print(
                        f"BUY: "
                        f"{probabilities[1] * 100:.2f}%"
                    )


                    print(
                        f"SELL: "
                        f"{probabilities[2] * 100:.2f}%"
                    )


                    print(
                        f"Reason: {reason}"
                    )


                    print(
                        "=============================="
                    )


                    # ==================================
                    # OPTIONAL TRADE EXECUTION
                    # ==================================

                    if EXECUTE_TRADE:

                        current_position = (
                            get_current_position()
                        )


                        # ------------------------------
                        # BUY SIGNAL
                        # ------------------------------

                        if signal == "BUY":

                            if current_position is None:

                                print(
                                    "\nExecuting BUY..."
                                )

                                trade_result = buy()

                                print(
                                    trade_result
                                )


                            elif current_position == "SELL":

                                print(
                                    "\nSELL is open."
                                    " Exit SELL first."
                                )


                            else:

                                print(
                                    "\nBUY already open."
                                )


                        # ------------------------------
                        # SELL SIGNAL
                        # ------------------------------

                        elif signal == "SELL":

                            if current_position is None:

                                print(
                                    "\nExecuting SELL..."
                                )

                                trade_result = sell()

                                print(
                                    trade_result
                                )


                            elif current_position == "BUY":

                                print(
                                    "\nBUY is open."
                                    " Exit BUY first."
                                )


                            else:

                                print(
                                    "\nSELL already open."
                                )


                        # ------------------------------
                        # HOLD
                        # ------------------------------

                        else:

                            print(
                                "\nHOLD -> No trade executed."
                            )


        time.sleep(
            CHECK_INTERVAL
        )


# ==========================================
# STOP PROGRAM
# ==========================================

except KeyboardInterrupt:

    print(
        "\nBot stopped manually."
    )


finally:

    mt5.shutdown()

    print(
        "MT5 disconnected."
    )