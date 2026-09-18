import time
import joblib
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from models_assignment import gold_conv1d_model, gold_hybrid1_model,gold_hybrid2_model,US30_conv1d_model,US30_lstm_model,US30_hybrid1_model,US30_hybrid2_model
from pipeline import pipeline
from tradingfunc import getsymbol,get_candle_timer
from probablity_func import final_prediction2

c1d = []
h1 = []
h2 = []
lstm = []
# =========================================================
# SETTINGS
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SYMBOL1 = "GOLD"
SYMBOL2 = "US30Cash"
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
        _,features2 = getsymbol(SYMBOL2, SCALER_PATH,TIMEFRAME,112)
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

        g_c1d = [conv_hold, conv_buy, conv_sell]

        (
            hybrid1_prediction,
            hybrid1_hold,
            hybrid1_buy,
            hybrid1_sell
        ) = predict_model(
            gold_hybrid1_model,
            features
        )

        g_h1 = [hybrid1_hold, hybrid1_buy, hybrid1_sell]

        (
                    hybrid2_prediction,
                    hybrid2_hold,
                    hybrid2_buy,
                    hybrid2_sell
                ) = predict_model(
                    gold_hybrid2_model,
                    features
                )
        g_h2 = [hybrid2_hold, hybrid2_buy, hybrid2_sell]

        (
            US30_hybrid1_prediction,
            US30_hybrid1_hold,
            US30_hybrid1_buy,
            US30_hybrid1_sell
        ) = predict_model(
            US30_hybrid1_model,
            features
        )
          
        US30h2 = [US30_hybrid1_hold, US30_hybrid1_buy, US30_hybrid1_sell]

        (
                    US30_hybrid2_prediction,
                    US30_hybrid2_hold,
                    US30_hybrid2_buy,
                    US30_hybrid2_sell
                ) = predict_model(
                    US30_hybrid2_model,
                    features
                )
        US30h2 = [US30_hybrid2_hold,US30_hybrid2_buy,US30_hybrid2_sell]

        (
                    US30_conv_prediction,
                    US30_conv_hold,
                    US30_conv_buy,
                    US30_conv_sell
                ) = predict_model(
                    US30_conv1d_model,
                    features
                )
        # =================================================
        # MAJORITY VOTE
        # =================================================

        predictions = [
            conv_prediction,
            hybrid1_prediction,
            US30_hybrid1_prediction,
            US30_hybrid2_prediction,
            US30_conv_prediction
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
            f"Gold Conv1D : {conv_prediction:<5} | "
            f"HOLD {conv_hold * 100:6.2f}% | "
            f"BUY {conv_buy * 100:6.2f}% | "
            f"SELL {conv_sell * 100:6.2f}%"
        )

        print(
            f"Gold Hybrid1: {hybrid1_prediction:<5} | "
            f"HOLD {hybrid1_hold * 100:6.2f}% | "
            f"BUY {hybrid1_buy * 100:6.2f}% | "
            f"SELL {hybrid1_sell * 100:6.2f}%"
        )
        p = final_prediction2(g_c1d,g_h1)
        print("="*75)
        print(p)
        if p in ("BUY", "SELL"):

            with open("GOLd_record.txt", "a") as f:
                f.write(
                    f"{candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC \n"
                    "conv1d: "
                    f"HOLD {conv_hold * 100:6.2f}% | "
                    f"BUY {conv_buy * 100:6.2f}% | "
                    f"SELL {conv_sell * 100:6.2f}%\n"
                    "hybrid1: "
                    f"HOLD {hybrid1_hold * 100:6.2f}% | "
                    f"BUY {hybrid1_buy * 100:6.2f}% | "
                    f"SELL {hybrid1_sell * 100:6.2f}%\n\n"

                )

            print("Signal recorded to record.txt")
 
        time.sleep(1)


except KeyboardInterrupt:

    print("\n\nStopping program...")


finally:

    mt5.shutdown()

    print("MT5 disconnected.")