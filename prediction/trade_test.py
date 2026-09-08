import MetaTrader5 as mt5

print("Initializing MT5...")

if not mt5.initialize():
    print("Initialize FAILED:")
    print(mt5.last_error())
    quit()

print("Initialize OK")
print("Terminal:", mt5.terminal_info())
print("Account:", mt5.account_info())

rates = mt5.copy_rates_from_pos(
    "GOLD",
    mt5.TIMEFRAME_M15,
    1,
    100
)

print("Rates:", rates)

if rates is None:
    print("Rates FAILED:")
    print(mt5.last_error())
else:
    print("Rates received:", len(rates))

mt5.shutdown()