"""Step 1: data, time handling, indicators.  (Strategy rules are added in step 2.)"""
import numpy as np
import pandas as pd
import config as C

TF = pd.Timedelta(minutes=C.TIMEFRAME_MIN)

def effective_mode():
    """FUTURES data can never be LIVE."""
    return "TEST" if C.SOURCE_TYPE == "FUTURES" else C.MODE

def fetch_raw():
    import yfinance as yf        # imported here so tests run offline
    df = yf.download(C.SYMBOL, period=C.PERIOD, interval=C.INTERVAL, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def add_indicators(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for n in C.EMA_PERIODS:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr"] = wilder(tr, C.ATR_N)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    atr_adx = wilder(tr, C.ADX_N).replace(0, np.nan)
    d["plus_di"] = 100 * wilder(pdm, C.ADX_N) / atr_adx
    d["minus_di"] = 100 * wilder(mdm, C.ADX_N) / atr_adx
    s = (d["plus_di"] + d["minus_di"]).replace(0, np.nan)
    dx = (100 * (d["plus_di"] - d["minus_di"]).abs() / s).fillna(0.0)
    d["adx"] = wilder(dx, C.ADX_N)
    return d.replace([np.inf, -np.inf], np.nan)

def has_gap(df, window=None):
    """True if two consecutive candles inside the last `window` candles are > GAP_FACTOR x timeframe apart."""
    window = window or C.GAP_WINDOW
    idx = df.index[-window:]
    if len(idx) < 2:
        return False
    return bool(((idx[1:] - idx[:-1]) > C.GAP_FACTOR * TF).any())

def prepare(raw, now_utc=None):
    """Returns {"ok", "reason", "df", "meta"}. reason is None, "no_data" or "market_closed".
    df = closed candles only (UTC index) with indicators."""
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    meta = {"source_type": C.SOURCE_TYPE, "price_model": C.PRICE_MODEL, "source_tz": None,
            "n_closed": 0, "lag_candles": None, "gap_in_window": False, "note": None}

    def out(reason, df=None, note=None):
        meta["note"] = note
        return {"ok": reason is None, "reason": reason, "df": df, "meta": meta}

    if raw is None or len(raw) == 0:
        return out("no_data", note="empty")
    tz = getattr(raw.index, "tz", None)
    if tz is None:
        return out("no_data", note="timezone_unknown")      # unknown timezone -> never signal
    meta["source_tz"] = str(tz)
    try:
        df = raw[["Open", "High", "Low", "Close"]].astype(float)
    except Exception:
        return out("no_data", note="bad_columns")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert("UTC")
    df = df[df.index + TF <= now]                          # keep closed candles only
    meta["n_closed"] = int(len(df))
    if len(df) < C.MIN_CANDLES:
        return out("no_data", note=f"only_{len(df)}_candles")
    meta["lag_candles"] = round(float((now - (df.index[-1] + TF)) / TF), 2)
    if meta["lag_candles"] > C.MAX_LAG_CANDLES:
        return out("market_closed", note="stale_last_candle")
    df = add_indicators(df)
    meta["gap_in_window"] = has_gap(df)
    return out(None, df)


# =====================================================================
# Step 2: strategy rules (fixed, no tuning)
# =====================================================================
import json
import datetime as _dt
from zoneinfo import ZoneInfo

def _t(s):
    h, m = s.split(":")
    return _dt.time(int(h), int(m))

def load_news(path=None):
    """news_blackout.json: {"events": [{"time_utc": "2026-11-20T13:30:00Z", "name": "NFP"}]}"""
    try:
        with open(path or C.NEWS_FILE) as f:
            data = json.load(f)
        out = []
        for e in data.get("events", []):
            ts = pd.Timestamp(e["time_utc"])
            out.append(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))
        return sorted(out)
    except Exception:
        return []

def news_file_stale(news, now_utc):
    now = pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return not any(now <= e <= now + pd.Timedelta(days=C.NEWS_STALE_DAYS) for e in (news or []))

def in_session(close_ts):
    """True if the signal-candle close time is inside London or New York session (DST aware)
    and not weekend / Friday after the cutoff."""
    u = close_ts.tz_convert("UTC")
    if u.weekday() >= 5:
        return False
    if u.weekday() == 4 and u.time() > _t(C.FRIDAY_CUTOFF_UTC):
        return False
    for tzname, a, b in C.SESSIONS:
        loc = close_ts.tz_convert(ZoneInfo(tzname)).time()
        if _t(a) <= loc <= _t(b):
            return True
    return False

def in_news_blackout(open_ts, close_ts, news):
    """Candle span (open..close) must not overlap any blackout window (touching edges is not overlap)."""
    ny = ZoneInfo("America/New_York")
    o_ny, c_ny = open_ts.tz_convert(ny), close_ts.tz_convert(ny)
    for d in {o_ny.date(), c_ny.date()}:
        if d.weekday() >= 5:
            continue
        for a, b in C.DAILY_BLACKOUTS_NY:
            ws = pd.Timestamp(_dt.datetime.combine(d, _t(a))).tz_localize(ny)
            we = pd.Timestamp(_dt.datetime.combine(d, _t(b))).tz_localize(ny)
            if open_ts < we and close_ts > ws:
                return True
    for ev in (news or []):
        ws = ev - pd.Timedelta(minutes=C.NEWS_BEFORE_MIN)
        we = ev + pd.Timedelta(minutes=C.NEWS_AFTER_MIN)
        if open_ts < we and close_ts > ws:
            return True
    return False

_NEEDED = ["Open", "High", "Low", "Close", "ema20", "ema50", "ema200", "atr", "adx", "plus_di", "minus_di"]

def evaluate_at(df, i, ctx=None, news=None):
    """Evaluate the strategy on candle i (candle S, already closed).
    ctx = {"open": {"BUY": bool, "SELL": bool},
           "last_signal_time": {"BUY": Timestamp|None, ...},   # close time of last signal of that side
           "last_sl_time": {"BUY": Timestamp|None, ...}}        # time SL was hit
    Only the side that matches the EMA200 position is evaluated (BUY above EMA200, SELL below)."""
    ctx = ctx or {}
    close_ts = df.index[i] + TF
    open_ts = df.index[i]

    def hold(reason, side=None):
        return {"status": "HOLD", "reason": reason, "side": side, "time_utc": close_ts.isoformat()}

    # --- 1) no_data (also: gap in S-6..S, NaN, degenerate ATR)
    if i + 1 < C.MIN_CANDLES or i - 7 < 0:
        return hold("no_data")
    if has_gap(df.iloc[: i + 1]) or df.iloc[i - 7: i + 1][_NEEDED].isna().any().any():
        return hold("no_data")
    r = df.iloc[i]
    atr = float(r["atr"])
    if atr <= 0:
        return hold("no_data")
    # (market_closed is decided on the live data in prepare(); it does not apply to past candles)
    # --- 3) session, 4) news, 5) volatility
    if not in_session(close_ts):
        return hold("no_session")
    if in_news_blackout(open_ts, close_ts, news):
        return hold("news_blackout")
    rng = float(r["High"] - r["Low"])
    if rng > C.RANGE_MAX_ATR * atr:
        return hold("volatility_spike")

    c, o = float(r["Close"]), float(r["Open"])
    side = "BUY" if c > float(r["ema200"]) else "SELL"
    buy = side == "BUY"

    # --- 6) open signal, 7) cooldown
    if (ctx.get("open") or {}).get(side):
        return hold("open_signal", side)
    sl_time = (ctx.get("last_sl_time") or {}).get(side)
    if sl_time is not None:
        k = (close_ts - sl_time) / TF
        if 0 <= k <= C.COOLDOWN_CANDLES:
            return hold("cooldown", side)

    # --- 8) weak_trend
    e50, e200 = float(r["ema50"]), float(r["ema200"])
    e50_prev = float(df["ema50"].iloc[i - C.SLOPE_LOOKBACK])
    if buy:
        trend_ok = c > e200 and e50 > e200 and e50 > e50_prev
    else:
        trend_ok = c < e200 and e50 < e200 and e50 < e50_prev
    if not trend_ok:
        return hold("weak_trend", side)

    # --- 9) low_adx
    pdi, mdi = float(r["plus_di"]), float(r["minus_di"])
    dir_ok = pdi > mdi if buy else mdi > pdi
    if not (float(r["adx"]) >= C.ADX_MIN and dir_ok):
        return hold("low_adx", side)

    # --- 10) pullback
    lo = i - C.PULLBACK_MAX
    seg = df.iloc[lo:i]                                    # the 5 candles before S
    mask = (seg["Low"] <= seg["ema20"]) if buy else (seg["High"] >= seg["ema20"])
    if not mask.any():
        return hold("no_pullback", side)
    tpos = lo + int(np.argmax(mask.values))                # FIRST touch candle
    touch_open = df.index[tpos]
    last_sig = (ctx.get("last_signal_time") or {}).get(side)
    if last_sig is not None and touch_open < last_sig:
        return hold("no_pullback", side)
    zone = df.iloc[tpos:i]                                 # touch candle .. candle before S
    bad = (zone["Close"] < zone["ema50"]).any() if buy else (zone["Close"] > zone["ema50"]).any()
    if bad:
        return hold("no_pullback", side)

    # --- 11) trigger
    body = abs(c - o)
    body_ok = rng > 0 and body >= C.BODY_MIN * rng
    if buy:
        trig = c > float(zone["High"].max()) and c > o and c > float(r["ema20"]) and body_ok
    else:
        trig = c < float(zone["Low"].min()) and c < o and c < float(r["ema20"]) and body_ok
    if not trig:
        return hold("no_trigger", side)

    # --- 12) SL / TP (spread counted once)
    span = df.iloc[tpos:i + 1]
    if buy:
        entry = c + C.SPREAD_USD                           # Ask
        sl = float(span["Low"].min()) - C.SL_BUFFER_ATR * atr
        R = entry - sl
    else:
        entry = c                                          # Bid
        sl = float(span["High"].max()) + C.SL_BUFFER_ATR * atr
        R = sl - entry
    if R < C.R_MIN_ATR * atr:
        R = C.R_MIN_ATR * atr
        sl = entry - R if buy else entry + R
    if R > C.R_MAX_ATR * atr:
        return hold("sl_too_wide", side)
    tp = entry + C.RR * R if buy else entry - C.RR * R
    return {"status": "SIGNAL", "reason": None, "side": side, "time_utc": close_ts.isoformat(),
            "reference_price": round(c, 2), "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "r": round(R, 2), "sl_distance": round(R, 2), "tp_distance": round(C.RR * R, 2),
            "atr": round(atr, 2), "adx": round(float(r["adx"]), 2), "above_ema200": bool(c > e200),
            "touch_time_utc": touch_open.isoformat(), "source_type": C.SOURCE_TYPE}

def check_hit(sig, candle):
    """Closed-candle hit test. BUY is checked on Bid, SELL on Ask (= price + spread).
    If both SL and TP are touched inside one candle, SL wins."""
    sp = C.SPREAD_USD
    if sig["side"] == "BUY":
        sl_hit, tp_hit = candle["Low"] <= sig["sl"], candle["High"] >= sig["tp"]
    else:
        sl_hit, tp_hit = candle["High"] + sp >= sig["sl"], candle["Low"] + sp <= sig["tp"]
    if sl_hit:
        return "SL"
    if tp_hit:
        return "TP"
    return None
