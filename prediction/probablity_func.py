def get_signal(conv, h1, h2):
    """
    Input format:
    conv = [HOLD, BUY, SELL]
    h1   = [HOLD, BUY, SELL]
    h2   = [HOLD, BUY, SELL]

    Returns:
        BUY / SELL / HOLD
    """

    c_hold, c_buy, c_sell = conv
    h1_hold, h1_buy, h1_sell = h1
    h2_hold, h2_buy, h2_sell = h2

    # Weighted ensemble:
    # Hybrid2 ko zyada weight because directional moves
    # tumhare data me is model me zyada strongly appear ho rahe hain.
    avg_hold = 0.20*c_hold + 0.30*h1_hold + 0.50*h2_hold
    avg_buy  = 0.20*c_buy  + 0.30*h1_buy  + 0.50*h2_buy
    avg_sell = 0.20*c_sell + 0.30*h1_sell + 0.50*h2_sell

    # =========================================================
    # STRONG SELL
    # =========================================================
    if (
        h2_sell >= 48
        and h2_hold <= 15
    ):
        return "SELL"

    # H2 bearish + H1 HOLD weak
    elif (
        h2_sell >= 42
        and h2_sell > h2_buy
        and h2_hold <= 20
        and h1_hold < 39
    ):
        return "SELL"

    # Multiple models leaning SELL
    elif (
        c_sell >= c_buy
        and h2_sell > h2_buy
        and avg_sell >= 38
    ):
        return "SELL"


    # =========================================================
    # STRONG BUY
    # =========================================================
    elif (
        h2_buy >= 37
        and h2_buy > h2_sell
        and h2_hold <= 30
        and c_buy >= 37
    ):
        return "BUY"

    # Conv + Hybrid2 agree on BUY
    elif (
        c_buy > c_sell
        and h2_buy > h2_sell
        and c_buy >= 35
        and h2_buy >= 35
    ):
        return "BUY"

    # Overall ensemble bullish
    elif (
        avg_buy > avg_sell + 3
        and avg_buy > avg_hold
    ):
        return "BUY"


    # =========================================================
    # HOLD / UNCERTAINTY
    # =========================================================
    elif (
        avg_hold >= 38
        and avg_hold > avg_buy
        and avg_hold > avg_sell
    ):
        return "HOLD"

    # BUY and SELL too close = indecision
    elif abs(avg_buy - avg_sell) <= 2.5:
        return "HOLD"

    # Models fighting each other
    elif (
        h1_hold >= 39
        and abs(h2_buy - h2_sell) <= 5
    ):
        return "HOLD"

    else:
        return "HOLD"


def predict_signal(conv, h1, h2):

    ch, cb, cs = conv
    h1h, h1b, h1s = h1
    h2h, h2b, h2s = h2

    # Direction differences
    conv_diff = cb - cs
    h1_diff = h1b - h1s
    h2_diff = h2b - h2s

    # -----------------------------------
    # PURE / STRONG SELL
    # -----------------------------------

    if h2s >= 48 and h2h <= 15:
        return "SELL"

    if (
        h2s >= 42
        and h2_diff <= -2
        and h1h < 39
    ):
        return "SELL"


    # -----------------------------------
    # BUY
    # -----------------------------------

    if (
        h2h <= 30
        and h2b >= 37
        and h2_diff >= 1
        and cb >= cs
    ):
        return "BUY"


    # -----------------------------------
    # HOLD
    # -----------------------------------

    if (
        ch >= 40
        and h1h >= 40
    ):
        return "HOLD"

    if (
        abs(h2_diff) < 2
        and abs(h1_diff) < 3
    ):
        return "HOLD"

    if h1h >= 40:
        return "HOLD"


    # -----------------------------------
    # FALLBACK ensemble
    # -----------------------------------

    buy_score = (
        cb * 0.20 +
        h1b * 0.30 +
        h2b * 0.50
    )

    sell_score = (
        cs * 0.20 +
        h1s * 0.30 +
        h2s * 0.50
    )

    hold_score = (
        ch * 0.20 +
        h1h * 0.30 +
        h2h * 0.50
    )

    if buy_score > sell_score + 3 and buy_score > hold_score:
        return "BUY"

    elif sell_score > buy_score + 3 and sell_score > hold_score:
        return "SELL"

    return "HOLD"