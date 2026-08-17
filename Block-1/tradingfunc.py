import MetaTrader5 as mt5
import threading
import time

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

mt5.initialize()

running = True

mt5.shutdown()