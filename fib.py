"""Setup B - Fibonacci pullback inside a swing trend. Rule-based, causal (a swing is used only after it is confirmed).
UPTREND : last swings  low1, high1, low2, high2  with low2 > low1 and high2 > high1  -> leg = low2 -> high2.
DOWNTREND: last swings high1, low1, high2, low2  with high2 < high1 and low2 < low1  -> leg = high2 -> low2.
Entry   : price corrects 38.2% - 78.6% of the leg, then a CLOSED reversal candle confirms (see evaluate_fib_at)."""
import numpy as np
import pandas as pd
import config as C
import strategy as S

TF = S.TF
_NEEDED = ["Open", "High", "Low", "Close", "ema20", "ema200", "atr"]

def swing_pivots(w, L):
    """Confirmed fractal pivots of window w -> [(pos, 'H'|'L', price)].
    pivot high at k: High[k] > the L highs before it and >= the L highs after it (mirror for lows).
    It needs L candles to its right, so the newest L candles can never be a pivot (no look-ahead)."""
    H, Lo = w["High"].values, w["Low"].values
    n = len(w)
    if n < 2 * L + 1:
        return []
    from numpy.lib.stride_tricks import sliding_window_view as swv
    hmax, lmin = swv(H, L).max(axis=1), swv(Lo, L).min(axis=1)     # hmax[j] = max(H[j:j+L])
    ks = np.arange(L, n - L)
    is_h = (H[ks] > hmax[ks - L]) & (H[ks] >= hmax[ks + 1])
    is_l = (Lo[ks] < lmin[ks - L]) & (Lo[ks] <= lmin[ks + 1])
    out = []
    for k, h, l in zip(ks, is_h, is_l):
        if h and l:
            continue                                               # one huge candle is both: ambiguous, skip
        if h:
            out.append((int(k), "H", float(H[k])))
        elif l:
            out.append((int(k), "L", float(Lo[k])))
    return out

def alternate(piv):
    """Make highs and lows alternate: two highs in a row -> keep the higher; two lows in a row -> keep the lower."""
    out = []
    for k, t, p in piv:
        if out and out[-1][1] == t:
            if (t == "H" and p > out[-1][2]) or (t == "L" and p < out[-1][2]):
                out[-1] = (k, t, p)
        else:
            out.append((k, t, p))
    return out

def structure(alt):
    """-> (side, leg_start, leg_end) or None. If both an up and a down structure exist, the one ending later wins."""
    found = []
    for t in ("H", "L"):
        idx = [j for j, x in enumerate(alt) if x[1] == t]
        if not idx or idx[-1] < 3:
            continue
        p0, p1, p2, p3 = alt[idx[-1] - 3: idx[-1] + 1]
        if t == "H" and p2[2] > p0[2] and p3[2] > p1[2]:           # higher low, higher high
            found.append((p3[0], "BUY", p2, p3))
        if t == "L" and p2[2] < p0[2] and p3[2] < p1[2]:           # lower high, lower low
            found.append((p3[0], "SELL", p2, p3))
    if not found:
        return None
    found.sort(key=lambda x: x[0])
    return found[-1][1:]

def evaluate_fib_at(df, i, ctx=None, news=None):
    """HOLD reason order: no_data, no_session, volatility_spike, no_structure, open_signal, cooldown, small_leg,
    no_correction, deep_correction, stale_turn, no_reversal, sl_too_wide."""
    ctx = ctx or {}
    close_ts, open_ts = df.index[i] + TF, df.index[i]

    def hold(reason, side=None):
        return {"status": "HOLD", "reason": reason, "side": side, "time_utc": close_ts.isoformat(), "setup": "B"}

    if i + 1 < C.MIN_CANDLES or i - 7 < 0:
        return hold("no_data")
    if S.has_gap(df.iloc[: i + 1]) or df.iloc[i - 7: i + 1][_NEEDED].isna().any().any():
        return hold("no_data")
    r = df.iloc[i]
    atr = float(r["atr"])
    if atr <= 0:
        return hold("no_data")
    if not S.in_session(close_ts):
        return hold("no_session")
    rng = float(r["High"] - r["Low"])
    if rng > C.RANGE_MAX_ATR * atr:
        return hold("volatility_spike")

    w = df.iloc[max(0, i - C.FIB_LOOKBACK + 1): i + 1]
    st = structure(alternate(swing_pivots(w, C.FIB_PIVOT_L)))
    if st is None:
        return hold("no_structure")
    side, A, B = st
    buy = side == "BUY"
    if (ctx.get("open") or {}).get(side):
        return hold("open_signal", side)
    sl_time = (ctx.get("last_sl_time") or {}).get(side)
    if sl_time is not None and 0 <= (close_ts - sl_time) / TF <= C.COOLDOWN_CANDLES:
        return hold("cooldown", side)

    leg = abs(B[2] - A[2])
    if leg < C.FIB_LEG_MIN_ATR * atr:
        return hold("small_leg", side)

    after = w.iloc[B[0] + 1:]                              # candles after the leg end, S included
    if buy:
        if float(after["High"].max()) > B[2]:              # a new high already formed: no pullback to trade
            return hold("no_correction", side)
        extreme = float(after["Low"].min()); epos = B[0] + 1 + int(np.argmin(after["Low"].values))
        depth, broke = (B[2] - extreme) / leg, extreme <= A[2]
    else:
        if float(after["Low"].min()) < B[2]:
            return hold("no_correction", side)
        extreme = float(after["High"].max()); epos = B[0] + 1 + int(np.argmax(after["High"].values))
        depth, broke = (extreme - B[2]) / leg, extreme >= A[2]
    if broke or depth > C.FIB_MAX:
        return hold("deep_correction", side)
    if depth < C.FIB_MIN:
        return hold("no_correction", side)
    last_sig = (ctx.get("last_signal_time") or {}).get(side)
    if last_sig is not None and w.index[epos] < last_sig:  # this correction already produced a signal
        return hold("no_correction", side)
    if (len(w) - 1) - epos > C.FIB_FRESH:
        return hold("stale_turn", side)

    c, o, prev = float(r["Close"]), float(r["Open"]), w.iloc[-2]
    body_ok = rng > 0 and abs(c - o) >= C.BODY_MIN * rng
    if buy:
        ok = c > o and c > float(prev["High"]) and c > float(r["ema20"]) and c > float(r["ema200"]) and body_ok
    else:
        ok = c < o and c < float(prev["Low"]) and c < float(r["ema20"]) and c < float(r["ema200"]) and body_ok
    if not ok:
        return hold("no_reversal", side)

    if buy:
        entry = c + C.SPREAD_USD                           # Ask
        sl = extreme - C.SL_BUFFER_ATR * atr
        R = entry - sl
    else:
        entry = c                                          # Bid
        sl = extreme + C.SL_BUFFER_ATR * atr
        R = sl - entry
    if R < C.R_MIN_ATR * atr:
        R = C.R_MIN_ATR * atr
        sl = entry - R if buy else entry + R
    if R > C.R_MAX_ATR * atr:
        return hold("sl_too_wide", side)
    tp = entry + C.RR * R if buy else entry - C.RR * R
    adx = float(r["adx"]) if "adx" in r.index and np.isfinite(r["adx"]) else None
    return {"status": "SIGNAL", "reason": None, "side": side, "setup": "B", "time_utc": close_ts.isoformat(),
            "reference_price": round(c, 2), "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "r": round(R, 2), "sl_distance": round(R, 2), "tp_distance": round(C.RR * R, 2),
            "atr": round(atr, 2), "adx": None if adx is None else round(adx, 2),
            "above_ema200": bool(c > float(r["ema200"])), "touch_time_utc": w.index[epos].isoformat(),
            "fib_depth": round(depth, 3), "leg_start": round(A[2], 2), "leg_end": round(B[2], 2),
            "source_type": C.SOURCE_TYPE}
