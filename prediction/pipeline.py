import pandas as pd
import datetime as dt
import numpy as np
def pipeline(rates):

    # -----------------------------
    # RAW MT5 DATA → DATAFRAME
    # -----------------------------

    df = pd.DataFrame(rates)

    df = df.drop(
        columns=["real_volume"],
        errors="ignore"
    )

    # -----------------------------
    # TIME
    # -----------------------------

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute

    df["time_slot"] = (
        df["hour"] * 4 +
        (df["minute"] // 15)
    )

    df["time_slot"] = df["time_slot"] / 95

    # -----------------------------
    # CANDLE FEATURES
    # -----------------------------

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body"] = (
        df["close"] - df["open"]
    )

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df[["open", "close"]].min(axis=1)
    )

    # FIX
    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )
    

    # -----------------------------
    # TRUE RANGE / ATR
    # -----------------------------

    prev_close = df["close"].shift(1)

    df["tr"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr_14"] = (
        df["tr"].rolling(14).mean()
    )
    df["upper_wick_ratio"] = df["upper_wick"] / df["atr_14"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["atr_14"]
    
    
    df["body_ratio"] = abs(df["body"]) / df["range"]
  
    
    df["price_change_pct"] = (
        df["close"].pct_change() * 100
    )

   

    

    # -----------------------------
    # LAG FEATURES
    # -----------------------------

    features = [
        "open",
        "high",
        "low",
        "close",
        "tick_volume"
    ]

    for feature in features:

        for lag in range(1, 6):

            df[f"{feature}_diff_{lag}"] = (
                df[feature]
                - df[feature].shift(lag)
            )
    # Returns
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

# Momentum
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)


    # SMA
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

# EMA
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    
    df["close_sma5_dist"] = (
    (df["close"] - df["sma_5"]) / df["sma_5"]
)

    df["close_sma10_dist"] = (
    (df["close"] - df["sma_10"]) / df["sma_10"]
)

    df["close_sma20_dist"] = (
    (df["close"] - df["sma_20"]) / df["sma_20"]
)

    df["close_ema5_dist"] = (
    (df["close"] - df["ema_5"]) / df["ema_5"]
)

    df["close_ema10_dist"] = (
    (df["close"] - df["ema_10"]) / df["ema_10"]
)

    df["close_ema20_dist"] = (
    (df["close"] - df["ema_20"]) / df["ema_20"]
)
    df["ema_10_slope"] = (
    df["ema_10"] - df["ema_10"].shift(3)
)

    df["ema_20_slope"] = (
    df["ema_20"] - df["ema_20"].shift(3)
)

    df["ema_10_slope_pct"] = (
    (df["ema_10"] - df["ema_10"].shift(3))
    / df["ema_10"].shift(3)
)

    df["ema_20_slope_pct"] = (
    (df["ema_20"] - df["ema_20"].shift(3))
    / df["ema_20"].shift(3)
)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for period in [7, 14, 21]:

       avg_gain = gain.rolling(period).mean()
       avg_loss = loss.rolling(period).mean()

       rs = avg_gain / (avg_loss + 1e-8)

       df[f"rsi_{period}"] = 100 - (
        100 / (1 + rs))

    ema_12 = df["close"].ewm(
    span=12,
    adjust=False
).mean()

    ema_26 = df["close"].ewm(
    span=26,
    adjust=False
).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
    span=9,
    adjust=False
).mean()

    df["macd_histogram"] = (
    df["macd"] - df["macd_signal"]
)

    df["atr_7"] = df["tr"].rolling(7).mean()

    df["atr_21"] = df["tr"].rolling(21).mean()

    df["atr_ratio"] = (
    df["atr_14"] / df["close"]
)
    df["day_of_week"] = df["time"].dt.dayofweek
    df["rolling_std_5"] = (
    df["close"].pct_change()
    .rolling(5)
    .std()
)

    df["rolling_std_10"] = (
    df["close"].pct_change()
    .rolling(10)
    .std()
)

    df["rolling_std_20"] = (
    df["close"].pct_change()
    .rolling(20)
    .std()
)

    candle_range = (
    df["high"] - df["low"] + 1e-8
)

    df["close_position"] = (
    (df["close"] - df["low"])
    / candle_range
)

    df["open_position"] = (
    (df["open"] - df["low"])
    / candle_range
)

    df["high_low_position"] = (
    (df["close"] - df["low"])
    / (df["high"] - df["low"] + 1e-8)
)

    
    df = df.dropna().reset_index(drop=True)
    df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

    df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

    df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

    df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)
    df["volume_ma_5"] = (
    df["tick_volume"].rolling(5).mean()
)

    df["volume_ma_20"] = (
    df["tick_volume"].rolling(20).mean()
)

    df["volume_ratio"] = (
    df["tick_volume"]
    / (df["volume_ma_20"] + 1e-8)
)
    df = df.drop(columns=['time', 'hour', 'minute','day_of_week'], errors='ignore')
    return df