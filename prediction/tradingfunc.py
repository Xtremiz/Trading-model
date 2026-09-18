import MetaTrader5 as mt5
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from pipeline import pipeline
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
running = True

def buy():
    tick = mt5.symbol_info_tick("GOLD")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "GOLD",
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 20,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }

    return mt5.order_send(request)

def exit_buy_trade():
    global running

    for pos in mt5.positions_get(symbol="GOLD") or []:
        if pos.type == mt5.POSITION_TYPE_BUY:

            tick = mt5.symbol_info_tick("GOLD")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "GOLD",
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL,
                "position": pos.ticket,
                "price": tick.bid,
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }

            print("\n", mt5.order_send(request))

    running = False

def sell():
    tick = mt5.symbol_info_tick("GOLD")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "GOLD",
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_SELL,
        "price": tick.bid,
        "deviation": 20,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }

    return mt5.order_send(request)

def exit_sell_trade():
    global running

    for pos in mt5.positions_get(symbol="GOLD") or []:
        if pos.type == mt5.POSITION_TYPE_SELL:

            tick = mt5.symbol_info_tick("GOLD")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": "GOLD",
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": tick.ask,
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }

            print("\n", mt5.order_send(request))

    running = False
# _SCALER_CACHE = {}
# def _load_scaler(scaler_path):

#     if scaler_path not in _SCALER_CACHE:

#         try:
#             _SCALER_CACHE[scaler_path] = joblib.load(scaler_path)
#             print(f"Scaler loaded: {scaler_path}")

#         except Exception as e:
#             print(f"Scaler load failed ({scaler_path}): {e}")
#             return None

#     return _SCALER_CACHE[scaler_path]

def getsymbol(symbol, scaler_path, timeframe, NUM_FEATURES):

    try:
        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)
 
    except Exception as e:
        print(f"Scaler load failed ({scaler_path}): {e}")
        return None, None
 
    # -----------------------------------------------------
    # 2. Rates
    # -----------------------------------------------------
 
    SEQUENCE_LENGTH = 100
 
    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        1,
        150
    )
 
    if rates is None:
        print(f"MT5 data nahi mili ({symbol}):", mt5.last_error())
        return None, None
 
    # -----------------------------------------------------
    # 3. DataFrame
    # -----------------------------------------------------
 
    df = pd.DataFrame(rates)
 
    if df.empty:
        print(f"Empty dataframe: {symbol}")
        return None, None
 
    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )
 
    candle_time = df["time"].iloc[-1]
 
    # -----------------------------------------------------
    # 4. Pipeline
    # -----------------------------------------------------
 
    features = pipeline(df)
 
    if features is None:
        print(f"Pipeline failed: {symbol}")
        return candle_time, None
 
    # -----------------------------------------------------
    # 5. NumPy
    # -----------------------------------------------------
 
    features = np.asarray(
        features,
        dtype=np.float32
    )
 
    if features.ndim == 1:
        features = features.reshape(1, -1)
 
    # -----------------------------------------------------
    # 6. Sequence length check
    # -----------------------------------------------------
 
    if len(features) < SEQUENCE_LENGTH:
        print(
            f"\nNot enough features ({symbol}): "
            f"{len(features)}/{SEQUENCE_LENGTH}"
        )
        return candle_time, None
 
    features = features[-SEQUENCE_LENGTH:]
 
    # -----------------------------------------------------
    # 7. Feature count check
    # -----------------------------------------------------
 
    if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
        print(
            f"\nFeature mismatch ({symbol})! "
            f"Expected: {NUM_FEATURES}, "
            f"Got: {features.shape[1] if features.ndim == 2 else features.shape}"
        )
        return candle_time, None
 
    # -----------------------------------------------------
    # 8. Scale using saved RobustScaler
    # -----------------------------------------------------
 
    try:
        features = scaler.transform(features)
 
    except Exception as e:
        print(f"Scaler transform failed ({symbol}): {e}")
        return candle_time, None
 
    # -----------------------------------------------------
    # 9. Convert to torch.float32, add batch dim, move to DEVICE
    #    -> ready to pass directly into a model: model(features)
    # -----------------------------------------------------
 
    features = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)
 
    return candle_time, features
    # -----------------------------------------------------
    # 1. Scaler (cached)
    # -----------------------------------------------------

    scaler = _load_scaler(scaler_path)

    if scaler is None:
        return None, None

    # -----------------------------------------------------
    # 2. Rates
    # -----------------------------------------------------
    SEQUENCE_LENGTH = 100
    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        1,
        150
    )

    if rates is None:
        print(f"MT5 data nahi mili ({symbol}):", mt5.last_error())
        return None, None

    # -----------------------------------------------------
    # 3. DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(rates)

    if df.empty:
        print(f"Empty dataframe: {symbol}")
        return None, None

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    candle_time = df["time"].iloc[-1]

    # -----------------------------------------------------
    # 4. Pipeline
    # -----------------------------------------------------

    features = pipeline(df)

    if features is None:
        print(f"Pipeline failed: {symbol}")
        return candle_time, None

    # -----------------------------------------------------
    # 5. NumPy
    # -----------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.ndim == 1:
        features = features.reshape(1, -1)

    # -----------------------------------------------------
    # 6. Sequence length check
    # -----------------------------------------------------

    if len(features) < SEQUENCE_LENGTH:
        print(
            f"\nNot enough features ({symbol}): "
            f"{len(features)}/{SEQUENCE_LENGTH}"
        )
        return candle_time, None

    features = features[-SEQUENCE_LENGTH:]

    # -----------------------------------------------------
    # 7. Feature count check
    # -----------------------------------------------------

    if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
        print(
            f"\nFeature mismatch ({symbol})! "
            f"Expected: {NUM_FEATURES}, "
            f"Got: {features.shape[1] if features.ndim == 2 else features.shape}"
        )
        return candle_time, None

    # -----------------------------------------------------
    # 8. Scale using saved RobustScaler
    # -----------------------------------------------------

    try:
        features = scaler.transform(features)

    except Exception as e:
        print(f"Scaler transform failed ({symbol}): {e}")
        return candle_time, None

    # -----------------------------------------------------
    # 9. Convert to torch.float32, add batch dim, move to DEVICE
    #    -> ready to pass directly into a model: model(features)
    # -----------------------------------------------------

    features = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    return candle_time, features

def get_candle_timer():

    """
    Returns:
        current candle time
        remaining MM:SS
    """

    now = pd.Timestamp.now(tz="UTC")

    current_candle = now.floor("15min")

    next_candle = current_candle + pd.Timedelta(minutes=15)

    remaining = next_candle - now

    total_seconds = max(0, int(remaining.total_seconds()))

    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return (
        current_candle,
        f"{minutes:02d}:{seconds:02d}"
    )

mt5.initialize()

running = True

mt5.shutdown()