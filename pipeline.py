import pandas as pd


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

    # -----------------------------
    # PRICE CHANGE
    # -----------------------------

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
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=['time', 'hour', 'minute'], errors='ignore')
    return df