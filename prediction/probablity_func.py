def final_prediction2(conv1d, hybrid1):

    def normalize(x):
        x = list(map(float, x))
        if max(x) <= 1.0:
            x = [v * 100.0 for v in x]
        return x

    def score(p):
        """
        2p-1 transform, scaled to -100..+100.
        50% -> 0, 100% -> +100, 0% -> -100
        """
        frac = p / 100.0
        return (2 * frac - 1) * 100.0

    conv1d = normalize(conv1d)
    hybrid1 = normalize(hybrid1)

    c_hold_raw, c_buy_raw, c_sell_raw = conv1d
    h_hold_raw, h_buy_raw, h_sell_raw = hybrid1

    # 2p-1 scaled scores (-100 to +100)
    c_hold = score(c_hold_raw)
    c_buy  = score(c_buy_raw)
    c_sell = score(c_sell_raw)

    h_hold = score(h_hold_raw)
    h_buy  = score(h_buy_raw)
    h_sell = score(h_sell_raw)

    # =========================================================
    # RULE 1 (highest priority):
    # Conv1D 99% HOLD -> score(99) = 98
    # =========================================================
    if c_hold >= 98:
        return "HOLD"

    # =========================================================
    # RULE 2:
    # Hybrid1 SELL >= 90% -> score(90) = 80
    # =========================================================
    if h_sell >= 80:
        return "SELL"

    # =========================================================
    # RULE 3:
    # BUY ab teen cheezon pe depend karta hai:
    #   a) Conv1D majority BUY ho (buy > hold aur buy > sell)
    #   b) Conv1D buy score bhi khud ek minimum threshold cross kare
    #      (e.g. c_buy_raw >= 20% -> score(20) = -60)
    #   c) Hybrid1 BUY >= 34% -> score(34) = -32
    # =========================================================
    c_majority_buy = (c_buy > c_hold) and (c_buy > c_sell)

    C_BUY_MIN = -60   # corresponds to c_buy_raw >= 20%

    if c_majority_buy and c_buy >= C_BUY_MIN and h_buy >= -32:
        return "BUY"

    # =========================================================
    # RULE 4:
    # Dono models ka majority HOLD ho => HOLD
    # =========================================================
    c_majority_hold = (c_hold > c_buy) and (c_hold > c_sell)
    h_majority_hold = (h_hold > h_buy) and (h_hold > h_sell)

    if c_majority_hold and h_majority_hold:
        return "HOLD"

    # =========================================================
    # FALLBACK
    # =========================================================
    return "HOLD"