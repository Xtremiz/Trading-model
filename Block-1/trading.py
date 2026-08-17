import MetaTrader5 as mt5
import pandas as pd
import threading
import time

from tradingfunc import (
    buy,
    sell,
    exit_buy_trade,
    exit_sell_trade
)


SYMBOL = "GOLD"
running = True


mt5.initialize()


def get_position():
    positions = mt5.positions_get(symbol=SYMBOL)

    if positions:
        return positions[0]

    return None


def live():
    while running:

        pos = get_position()
        tick = mt5.symbol_info_tick(SYMBOL)

        if tick:

            if pos:

                if pos.type == mt5.POSITION_TYPE_BUY:
                    trade_type = "BUY"
                else:
                    trade_type = "SELL"

                print(
                    f"\rBid: {tick.bid} | "
                    f"Ask: {tick.ask} | "
                    f"P/L: ${pos.profit:.2f} | "
                    f"Trade: {trade_type}",
                    end="",
                    flush=True
                )

            else:

                print(
                    f"\rBid: {tick.bid} | "
                    f"Ask: {tick.ask} | "
                    f"No open trade",
                    end="",
                    flush=True
                )

        time.sleep(0.2)


threading.Thread(
    target=live,
    daemon=True
).start()

def show_current_candle():

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        mt5.TIMEFRAME_M15,
        0,      # current forming candle
        1
    )

    if rates is None or len(rates) == 0:
        print("\nCurrent candle data unavailable")
        return

    candle = rates[0]

    candle_time = pd.to_datetime(
        candle["time"],
        unit="s"
    )

    print("\n" + "=" * 50)
    print("CURRENT FORMING CANDLE")
    print("=" * 50)

    print(f"Start Time  : {candle_time}")
    print(f"Open        : {candle['open']}")
    print(f"High        : {candle['high']}")
    print(f"Low         : {candle['low']}")
    print(f"Live Close  : {candle['close']}")
    print(f"Tick Volume : {candle['tick_volume']}")
    print(f"Spread      : {candle['spread']}")

    print("=" * 50)


while running:

    pos = get_position()

    # Koi trade open nahi
    if not pos:

        show_current_candle()

        command = input(
            "\nBUY / SELL\n>>> "
        ).lower().strip()


        if command == "buy":

            result = buy()

            print("\nBUY RESULT:")
            print(result)


        elif command == "sell":

            result = sell()

            print("\nSELL RESULT:")
            print(result)


        else:

            print("\nInvalid command")


    # Trade already open hai
    else:

        if pos.type == mt5.POSITION_TYPE_BUY:
            trade_type = "BUY"
        else:
            trade_type = "SELL"

        command = input(
            f"\n{trade_type} trade active. Type EXIT:\n>>> "
        ).lower().strip()


        if command == "exit":

            if pos.type == mt5.POSITION_TYPE_BUY:
                exit_buy_trade()

            elif pos.type == mt5.POSITION_TYPE_SELL:
                exit_sell_trade()