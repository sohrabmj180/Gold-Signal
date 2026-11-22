"""Setup C - Smart Money Concept: CHoCH -> higher low -> BOS -> Order Block retest (M15). Rule-based and causal.
BUY (SELL is the exact mirror image, done by flipping the prices):
  1. before: bearish structure  (lower high Hb < Ha, lower low Lb < La)
  2. CHoCH : a candle CLOSES above the last lower high Hb, making a swing high Hc > Hb
  3. HL    : the next swing low Lc > Lb
  4. BOS   : a candle CLOSES above Hc after Lc
  5. OB    : last strong bearish candle at the start of that leg (around Lc); at most SMC_OB_MAX_AGE candles old
  6. retest: price returns into the OB for the first time, and a strong bullish candle closes above the OB
  Also: daily bias must agree, high-volatility hours only, at most one setup-C signal per Tehran day."""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
import config as C
import strategy as S
import fib as F

TF = S.TF
TEHRAN = ZoneInfo("Asia/Tehran")
_NEEDED = ["Open", "High", "Low", "Close", "atr"]

def in_window(close_ts):
    """'High volatility hours' = C.SMC_SESSIONS (DST aware), start inclusive / end exclusive; weekends closed."""
    u = close_ts.tz_convert("UTC")
    if u.weekday() >= 5:
        return False
    if u.weekday() == 4 and u.time() > S._t(C.FRIDAY_CUTOFF_UTC):
        return False
    for tzname, a, b in C.SMC_SESSIONS:
        loc = close_ts.tz_convert(ZoneInfo(tzname)).time()
        if S._t(a) <= loc < S._t(b):
            return True
    return False

def daily_bias(df, i):
    """'BUY' if the last COMPLETED UTC day closed above the EMA of daily closes, 'SELL' if below, None if unknown."""
    day0 = df.index[i].normalize()
    lo, hi = df.index.searchsorted(day0 - pd.Timedelta(days=60)), df.index.searchsorted(day0)
    closes = df["Close"].iloc[lo:hi].resample("1D").last().dropna()
    if len(closes) < C.SMC_BIAS_EMA + 1:
        return None
    ema = closes.ewm(span=C.SMC_BIAS_EMA, adjust=False).mean()
    last, e = float(closes.iloc[-1]), float(ema.iloc[-1])
    return "BUY" if last > e else "SELL" if last < e else None

def _mirror(w):
    m = w[["Open", "High", "Low", "Close"]].copy()
    m["Open"], m["Close"], m["High"], m["Low"] = -w["Open"], -w["Close"], -w["Low"], -w["High"]
    return m

def find_structure(m, L):
    """Most recent CHoCH -> HL -> BOS pattern in BUY orientation, or None. Positions are relative to m."""
    alt = F.alternate(F.swing_pivots(m, L))
    cl = m["Close"].values
    for j in range(len(alt) - 1, 4, -1):
        if alt[j][1] != "L":
            continue
        Ha, La, Hb, Lb, Hc, Lc = alt[j - 5: j + 1]
        if not (Ha[1] == "H" and Hb[2] < Ha[2] and Lb[2] < La[2]):        # bearish structure before
            continue
        if not (Lc[2] > Lb[2]):                                            # higher low
            continue
        if not (Hc[2] > Hb[2] and (cl[Lb[0] + 1: Hc[0] + 1] > Hb[2]).any()):   # CHoCH: a CLOSE above the last lower high
            continue
        hit = np.flatnonzero(cl[Lc[0] + 1:] > Hc[2])                       # BOS: first CLOSE above Hc after Lc
        if len(hit) == 0:
            continue
        return {"Hb": Hb, "Hc": Hc, "Lc": Lc, "bos": Lc[0] + 1 + int(hit[0])}
    return None

def order_block(m, lc_pos, i_rel):
    """Last strong bearish candle at/just before the swing low that started the BOS leg (positions lc-3 .. lc+1)."""
    O, Cl, H, Lo = (m[c].values for c in ("Open", "Close", "High", "Low"))
    for k in range(min(i_rel, lc_pos + 1), max(0, lc_pos - 3) - 1, -1):
        rng = H[k] - Lo[k]
        if Cl[k] < O[k] and rng > 0 and (O[k] - Cl[k]) >= C.SMC_OB_BODY_MIN * rng:
            return k
    return None

def evaluate_smc_at(df, i, ctx=None, news=None):
    """HOLD reasons in order: no_data, no_session, volatility_spike, no_bias, daily_limit, open_signal, no_structure,
    no_ob, stale_ob, ob_broken, no_retest, ob_touched, no_reaction, sl_too_wide."""
    ctx = ctx or {}
    close_ts, open_ts = df.index[i] + TF, df.index[i]

    def hold(reason, side=None):
        return {"status": "HOLD", "reason": reason, "side": side, "time_utc": close_ts.isoformat(), "setup": "C"}

    if i + 1 < C.MIN_CANDLES or i - 7 < 0:
        return hold("no_data")
    if S.has_gap(df.iloc[: i + 1]) or df.iloc[i - 7: i + 1][_NEEDED].isna().any().any():
        return hold("no_data")
    r = df.iloc[i]
    atr = float(r["atr"])
    if atr <= 0:
        return hold("no_data")
    if not in_window(close_ts):
        return hold("no_session")
    rng = float(r["High"] - r["Low"])
    if rng > C.RANGE_MAX_ATR * atr:
        return hold("volatility_spike")
    side = daily_bias(df, i)
    if side is None:
        return hold("no_bias")
    buy = side == "BUY"
    if ctx.get("c_day") == close_ts.tz_convert(TEHRAN).date().isoformat():
        return hold("daily_limit", side)                   # one setup-C signal per day (so also: no new trade after a stop)
    if (ctx.get("open") or {}).get(side):
        return hold("open_signal", side)

    w = df.iloc[max(0, i - C.SMC_LOOKBACK + 1): i + 1]
    m = w[["Open", "High", "Low", "Close"]] if buy else _mirror(w)
    i_rel = len(w) - 1
    pat = find_structure(m, C.SMC_PIVOT_L)
    if pat is None:
        return hold("no_structure", side)
    if pat["bos"] >= i_rel:                                # the BOS candle is the signal candle itself: no retest yet
        return hold("no_retest", side)
    ob = order_block(m, pat["Lc"][0], i_rel)
    if ob is None:
        return hold("no_ob", side)
    if i_rel - ob > C.SMC_OB_MAX_AGE:
        return hold("stale_ob", side)
    O, H, Lo, Cl = (m[c].values for c in ("Open", "High", "Low", "Close"))
    ob_lo, ob_hi = float(Lo[ob]), float(H[ob])             # demand zone (in the buy-oriented frame)
    if (Cl[ob + 1: i_rel + 1] < ob_lo).any():
        return hold("ob_broken", side)
    touched = np.flatnonzero(Lo[pat["bos"] + 1: i_rel + 1] <= ob_hi)
    if len(touched) == 0:
        return hold("no_retest", side)
    first_touch = pat["bos"] + 1 + int(touched[0])
    if i_rel - first_touch > C.SMC_REACT_WINDOW:
        return hold("ob_touched", side)                    # the zone was already used earlier
    c, o, rg = float(Cl[i_rel]), float(O[i_rel]), float(H[i_rel] - Lo[i_rel])
    if not (c > o and c > ob_hi and rg > 0 and (c - o) >= C.SMC_REACTION_BODY * rg):
        return hold("no_reaction", side)

    c_orig = float(r["Close"])
    if buy:
        entry = c_orig + C.SPREAD_USD
        sl = ob_lo - C.SL_BUFFER_ATR * atr
        R = entry - sl
        zone_lo, zone_hi = ob_lo, ob_hi
        bos_level, choch_level = pat["Hc"][2], pat["Hb"][2]
    else:
        entry = c_orig
        zone_lo, zone_hi = -ob_hi, -ob_lo                  # supply zone in real prices
        sl = zone_hi + C.SL_BUFFER_ATR * atr
        R = sl - entry
        bos_level, choch_level = -pat["Hc"][2], -pat["Hb"][2]
    if R < C.R_MIN_ATR * atr:
        R = C.R_MIN_ATR * atr
        sl = entry - R if buy else entry + R
    if R > C.R_MAX_ATR * atr:
        return hold("sl_too_wide", side)
    tp = entry + C.RR * R if buy else entry - C.RR * R
    adx = float(r["adx"]) if "adx" in r.index and np.isfinite(r["adx"]) else None
    e200 = float(r["ema200"]) if "ema200" in r.index and np.isfinite(r["ema200"]) else None
    return {"status": "SIGNAL", "reason": None, "side": side, "setup": "C", "time_utc": close_ts.isoformat(),
            "reference_price": round(c_orig, 2), "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "r": round(R, 2), "sl_distance": round(R, 2), "tp_distance": round(C.RR * R, 2), "atr": round(atr, 2),
            "adx": None if adx is None else round(adx, 2),
            "above_ema200": None if e200 is None else bool(c_orig > e200),
            "ob_low": round(zone_lo, 2), "ob_high": round(zone_hi, 2), "ob_time_utc": w.index[ob].isoformat(),
            "bos_time_utc": w.index[pat["bos"]].isoformat(), "bos_level": round(bos_level, 2),
            "choch_level": round(choch_level, 2), "bias": side, "source_type": C.SOURCE_TYPE}
