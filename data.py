import MetaTrader5 as mt5
import pandas as pd
import pipeline as pip

from datetime import datetime, timedelta

if not mt5.initialize():
    print("MT5 initialize failed")
    quit()

print("MT5 connected successfully")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print("\nDownloading:")
print("From:", start_date)
print("To  :", end_date)

rates = mt5.copy_rates_range(
    "GOLD",
    mt5.TIMEFRAME_M15,
    start_date,
    end_date
)

if rates is None or len(rates) == 0:

    print("No data found")

    mt5.shutdown()
    quit()


print("\nRaw candles downloaded:", len(rates))

df = pd.DataFrame(rates)

# Convert Unix timestamp → datetime
df["time"] = pd.to_datetime(
    df["time"],
    unit="s"
)

df = pip.pipeline(df)

current_return = (
    (df["close"] - df["open"])
    / df["open"]
)


threshold = 0.001


df["signal"] = 0


df.loc[
    current_return > threshold,
    "signal"
] = 1


df.loc[
    current_return < -threshold,
    "signal"
] = -1


df["signal"] = df["signal"].shift(-1)


df = df.iloc[:-1]

df.to_csv(
    "golddata.csv",
    index=False
)

mt5.shutdown()

print("\nMT5 shutdown")