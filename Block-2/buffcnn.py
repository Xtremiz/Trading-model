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

# CNN ko 50 candles ki sequence chahiye
BUFFER_SIZE = 50

CHECK_INTERVAL = 1

MODEL_PATH = r"F:\Git-Hub\Trading model\models\best_cnn_model.pt"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# False = sirf prediction
# True = real trade execution
EXECUTE_TRADE = False


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
#
# IMPORTANT:
#
# Apna EXACT CNN architecture yahan paste karo.
#
# Class ka naam "mycnn" rakhna hai.
#
# Example:
#
# class mycnn(nn.Module):
#     def __init__(self, ...):
#         ...
#
#     def forward(self, x):
#         ...
#
# ==========================================

class MyCNN(nn.Module):
    def __init__(self, num_features=87):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1D
            nn.Conv1d(
                in_channels=num_features,
                out_channels=32,
                kernel_size=5
            ),

            # BatchNorm
            nn.BatchNorm1d(32),

            # Activation
            nn.ReLU(),

            # Conv1D
            nn.Conv1d(
                in_channels=32,
                out_channels=128,
                kernel_size=5
            ),

            # BatchNorm
            nn.BatchNorm1d(128),

            # Activation
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # CNN output -> 128
            nn.Linear(1536, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print("Using device:", DEVICE)


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

    print("\nModel loading failed:")
    print(e)

    raise SystemExit


# ==========================================
# LOAD MODEL METADATA
# ==========================================

# input_size agar checkpoint mein hai
# to use karo.
#
# Agar nahi hai to feature_names ki length
# automatically use hogi.

feature_names = checkpoint["feature_names"]

input_size = checkpoint.get(
    "input_size",
    len(feature_names)
)

class_mapping = checkpoint["class_mapping"]

confidence_threshold = checkpoint.get(
    "confidence_threshold",
    0.0
)


print("\nMODEL INFORMATION")

print("Input size:", input_size)

print("Number of features:", len(feature_names))

print("Sequence length:", BUFFER_SIZE)

print("Class mapping:", class_mapping)

print(
    "Confidence threshold:",
    confidence_threshold
)


# ==========================================
# CREATE MODEL
# ==========================================

model = mycnn()


# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

try:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

except Exception as e:

    print("\nMODEL WEIGHT LOADING ERROR")
    print(e)

    print(
        "\nMake sure mycnn architecture "
        "is EXACTLY the same as training."
    )

    raise SystemExit


model.to(DEVICE)

model.eval()

print("\nCNN model loaded successfully.")


# ==========================================
# MT5 INITIALIZATION
# ==========================================

if not mt5.initialize():

    print("MT5 initialization failed")

    print(
        mt5.last_error()
    )

    raise SystemExit


print("MT5 connected.")


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


# Convert dataframe rows to buffer

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
    # APPLY YOUR PIPELINE
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
    # TAKE ONLY REQUIRED FEATURES
    # --------------------------------------

    processed_df = processed_df[
        feature_names
    ]


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
    # CHECK SEQUENCE LENGTH
    # --------------------------------------

    if len(processed_df) < BUFFER_SIZE:

        print(
            "\nNOT ENOUGH VALID ROWS"
        )

        print(
            "Required:",
            BUFFER_SIZE
        )

        print(
            "Received:",
            len(processed_df)
        )

        return None


    # --------------------------------------
    # TAKE LAST 50 ROWS
    # --------------------------------------

    sequence_df = processed_df.iloc[
        -BUFFER_SIZE:
    ].copy()


    # --------------------------------------
    # DATAFRAME -> NUMPY
    # --------------------------------------

    X = sequence_df.to_numpy(
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
    # CHECK FEATURE COUNT
    # --------------------------------------

    if X.shape[1] != input_size:

        print(
            "\nINPUT SIZE MISMATCH"
        )

        print(
            "Expected features:",
            input_size
        )

        print(
            "Received features:",
            X.shape[1]
        )

        return None


    # --------------------------------------
    # CURRENT SHAPE
    #
    # X:
    #
    # (50, features)
    #
    # --------------------------------------

    print(
        "\nSequence shape:",
        X.shape
    )


    # --------------------------------------
    # NUMPY -> TENSOR
    # --------------------------------------

    X_tensor = torch.tensor(

        X,

        dtype=torch.float32,

        device=DEVICE

    )


    # --------------------------------------
    # CNN INPUT SHAPE
    #
    # Current:
    #
    # (50, features)
    #
    # Add batch:
    #
    # (1, 50, features)
    #
    # Then transpose:
    #
    # (1, features, 50)
    #
    # Conv1D normally expects:
    #
    # (batch, channels, sequence)
    #
    # --------------------------------------

    X_tensor = X_tensor.unsqueeze(0)

    X_tensor = X_tensor.transpose(
        1,
        2
    )


    print(
        "CNN input shape:",
        tuple(X_tensor.shape)
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
    # CONVERT TO PYTHON VALUES
    # --------------------------------------

    prediction = prediction.item()

    confidence = confidence.item()


    # --------------------------------------
    # CLASS -> SIGNAL
    # --------------------------------------

    signal = class_mapping[
        prediction
    ]


    # --------------------------------------
    # CONFIDENCE FILTER
    # --------------------------------------

    if confidence < confidence_threshold:

        print(
            f"Confidence "
            f"{confidence:.4f} "
            f"is below threshold "
            f"{confidence_threshold}"
        )

        signal = "HOLD"


    return {

        "signal": signal,

        "prediction": prediction,

        "confidence": confidence,

        "probabilities":
            probabilities.cpu().numpy(),

        "features": sequence_df

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

            candle_end = candle_start + pd.Timedelta(
                minutes=15
            )


            # Broker/server time

            now = pd.to_datetime(

                tick.time,

                unit="s"

            )


            remaining = candle_end - now


            remaining_seconds = max(

                0,

                int(
                    remaining.total_seconds()
                )

            )


            minutes = remaining_seconds // 60

            seconds = remaining_seconds % 60


            if position:

                positions = mt5.positions_get(
                    symbol=SYMBOL
                )


                profit = positions[0].profit


                trade_info = (

                    f"Trade: {position} | "

                    f"P/L: ${profit:.2f}"

                )

            else:

                trade_info = "No open trade"


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
            "\n\nCommand (buy / sell / exit): "
        ).lower().strip()


        position = get_current_position()


        # ----------------------------------
        # BUY
        # ----------------------------------

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


        # ----------------------------------
        # SELL
        # ----------------------------------

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


        # ----------------------------------
        # EXIT
        # ----------------------------------

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
# LIVE PREDICTION LOOP
# ==========================================

print(
    "\nLIVE CNN PREDICTION STARTED\n"
)


try:

    while True:

        latest = get_completed_candles(1)


        if latest is not None and not latest.empty:

            latest_row = latest.iloc[0].to_dict()

            latest_time = latest_row["time"]


            # ----------------------------------
            # NEW COMPLETED CANDLE DETECTED
            # ----------------------------------

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


                # Add completed candle to buffer

                buffer.append(
                    latest_row
                )


                last_candle_time = latest_time


                # ----------------------------------
                # PREDICTION
                # ----------------------------------

                result = predict()


                if result is None:

                    print(
                        "Prediction skipped."
                    )

                else:

                    signal = result["signal"]

                    confidence = result["confidence"]

                    probabilities = result[
                        "probabilities"
                    ]


                    print(
                        "\n========= CNN PREDICTION ========="
                    )


                    print(
                        "Signal:",
                        signal
                    )


                    print(
                        "Prediction:",
                        result["prediction"]
                    )


                    print(
                        "Confidence:",
                        f"{confidence * 100:.2f}%"
                    )


                    print(
                        "Probabilities:",
                        probabilities
                    )


                    print(
                        "Sequence:",
                        f"{BUFFER_SIZE} candles"
                    )


                    print(
                        "=================================="
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