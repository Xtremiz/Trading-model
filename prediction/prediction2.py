import time
import joblib
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from probablity_func import predict_signal, get_signal
import torch
import torch.nn as nn
from models_assignment import gold_conv1d_model, gold_hybrid1_model, gold_hybrid2_model
from pipeline import pipeline
from tradingfunc import getsymbol,get_candle_timer

c1d = []
h1 = []
h2 = []

# =========================================================
# SETTINGS
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SYMBOL1 = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_M15





SCALER_PATH = r"F:\Git-Hub\Trading model\data\gold112_robust_scaler.pkl"


# =========================================================
# MT5 INITIALIZATION
# =========================================================

if not mt5.initialize():

    print("MT5 initialize failed!")
    print("Error:", mt5.last_error())

    raise SystemExit


print("MT5 connected successfully!")


# =========================================================
# ACCOUNT INFORMATION
# =========================================================

account_info = mt5.account_info()

if account_info is not None:

    print("Account:", account_info.login)
    print("Server :", account_info.server)


# =========================================================
# SYMBOL CHECK
# =========================================================

if not mt5.symbol_select(SYMBOL1, True):

    print(f"Symbol select failed: {SYMBOL1}")
    print("MT5 Error:", mt5.last_error())

    mt5.shutdown()

    raise SystemExit


print(f"Symbol selected: {SYMBOL1}")


# =========================================================
# LABELS
# =========================================================

LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL"
}

def get_latest_candle_time(symbol):

    rates = mt5.copy_rates_from_pos(
        symbol,
        TIMEFRAME,
        1,
        1
    )

    if rates is None or len(rates) == 0:
        return None

    return pd.to_datetime(
        rates[-1]["time"],
        unit="s",
        utc=True
    )

def predict_model(model, x):
    """
    x is expected to already be a torch.float32 tensor of shape
    (1, SEQUENCE_LENGTH, NUM_FEATURES) on DEVICE (i.e. the output
    of getsymbol()) -- no further conversion is done here.
    """

    with torch.no_grad():

        output = model(x)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    hold_probability = probabilities[0].item()
    buy_probability = probabilities[1].item()
    sell_probability = probabilities[2].item()

    return (
        LABELS[prediction],
        hold_probability,
        buy_probability,
        sell_probability
    )

last_candle_time = None


try:

    while True:

        # =================================================
        # CHEAP CHECK: has a new candle closed?
        # =================================================

        candle_time = get_latest_candle_time(SYMBOL1)

        if candle_time is None:

            time.sleep(1)

            continue

        if candle_time == last_candle_time:

            # Still the same candle -> just show countdown,
            # do NOT run the heavy pipeline.

            current_candle, timer = get_candle_timer()

            print(
                f"\rCurrent Candle: "
                f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                f"Time Remaining: {timer}",
                end=""
            )

            time.sleep(1)

            continue

        last_candle_time = candle_time

        # =================================================
        # NEW CANDLE -> now run the heavy pipeline once
        # =================================================

        candle_time, features = getsymbol(SYMBOL1, SCALER_PATH,TIMEFRAME,112)

        if features is None:

            time.sleep(1)

            continue

        # =================================================
        # PREDICTIONS
        # =================================================

        (
            conv_prediction,
            conv_hold,
            conv_buy,
            conv_sell
        ) = predict_model(
            gold_conv1d_model,
            features
        )

        c1d = [conv_hold, conv_buy, conv_sell]

        (
            hybrid1_prediction,
            hybrid1_hold,
            hybrid1_buy,
            hybrid1_sell
        ) = predict_model(
            gold_hybrid1_model,
            features
        )

        h1 = [hybrid1_hold, hybrid1_buy, hybrid1_sell]

        (
            hybrid2_prediction,
            hybrid2_hold,
            hybrid2_buy,
            hybrid2_sell
        ) = predict_model(
            gold_hybrid2_model,
            features
        )

        h2 = [hybrid2_hold, hybrid2_buy, hybrid2_sell]

        # =================================================
        # MAJORITY VOTE
        # =================================================

        predictions = [
            conv_prediction,
            hybrid1_prediction,
            hybrid2_prediction
        ]

        # =================================================
        # TIMER
        # =================================================

        current_candle, timer = get_candle_timer()

        # =================================================
        # PRINT RESULT
        # =================================================

        print("\n\n" + "=" * 75)

        print(
            f"Candle : "
            f"{candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        print(
            f"Current Candle: "
            f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
            f"Time Remaining: {timer}"
        )

        print("-" * 75)

        print(
            f"Conv1D : {conv_prediction:<5} | "
            f"HOLD {conv_hold * 100:6.2f}% | "
            f"BUY {conv_buy * 100:6.2f}% | "
            f"SELL {conv_sell * 100:6.2f}%"
        )

        print(
            f"Hybrid1: {hybrid1_prediction:<5} | "
            f"HOLD {hybrid1_hold * 100:6.2f}% | "
            f"BUY {hybrid1_buy * 100:6.2f}% | "
            f"SELL {hybrid1_sell * 100:6.2f}%"
        )

        print(
            f"Hybrid2: {hybrid2_prediction:<5} | "
            f"HOLD {hybrid2_hold * 100:6.2f}% | "
            f"BUY {hybrid2_buy * 100:6.2f}% | "
            f"SELL {hybrid2_sell * 100:6.2f}%"
        )

        print("=" * 75)

        p1 = predict_signal(c1d, h1, h2)
        p2 = get_signal(c1d, h1, h2)

        print(
            f"Majority Vote: {p1} | Ensemble Signal: {p2}"
        )

        # -------------------------------------------------
        # Record BUY/SELL signals to record.txt
        # -------------------------------------------------

        if p1 in ("BUY", "SELL") or p2 in ("BUY", "SELL"):

            with open("record.txt", "a") as f:
                f.write(
                    f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                    f"Majority Vote: {p1} | Ensemble Signal: {p2}\n"
                )

            print("Signal recorded to record.txt")

        time.sleep(1)


except KeyboardInterrupt:

    print("\n\nStopping program...")


finally:

    mt5.shutdown()

    print("MT5 disconnected.")