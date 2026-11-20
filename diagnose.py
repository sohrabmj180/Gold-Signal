"""
diagnose.py - READ ONLY. Shows WHICH rule blocks the signals. Sends nothing, changes nothing.
Run from GitHub Actions (workflow "Diagnose"): needs the secret TD_KEY. Prints the report and writes diagnosis.txt.
It counts candles / signals only. It deliberately prints NO win rate or profit, so nothing can be tuned on hindsight.
"""
import os, sys
import numpy as np
import pandas as pd
import config as C
import strategy as S
import engine as E

TF = S.TF
SIDES = ("BUY", "SELL")
FUNNEL = ["no_data", "no_session", "volatility_spike", "weak_trend", "low_adx", "no_pullback", "no_trigger", "sl_too_wide"]
NAMES = {"no_data": "داده/گپ", "no_session": "خارج از ساعت کاری", "volatility_spike": "نوسان بیش‌ازحد کندل",
         "weak_trend": "روند", "low_adx": "ADX / جهت DI", "no_pullback": "پولبک", "no_trigger": "تریگر ورود",
         "sl_too_wide": "حد ضرر خیلی بزرگ"}

# ---------------------------------------------------------------- data
def clean_history(raw, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if getattr(raw.index, "tz", None) is None:
        raise ValueError("source timezone unknown")
    df = raw[["Open", "High", "Low", "Close"]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert("UTC")
    return df[df.index + TF <= now]

# ---------------------------------------------------------------- why did a candle fail?
def sub_causes(df, i, reason, side):
    """Which exact sub-condition(s) of the failing rule were not met (a rule can fail for several reasons)."""
    r = df.iloc[i]
    buy = side == "BUY"
    c, o = float(r["Close"]), float(r["Open"])
    out = []
    if reason == "weak_trend":
        e50, e200 = float(r["ema50"]), float(r["ema200"])
        prev = float(df["ema50"].iloc[i - C.SLOPE_LOOKBACK])
        if (e50 <= e200) if buy else (e50 >= e200):
            out.append("EMA50 در سمت اشتباهِ EMA200")
        if (e50 <= prev) if buy else (e50 >= prev):
            out.append("شیب EMA50 هم‌جهت نیست")
    elif reason == "low_adx":
        if float(r["adx"]) < C.ADX_MIN:
            out.append("ADX کمتر از حداقل")
        if not (float(r["plus_di"]) > float(r["minus_di"]) if buy else float(r["minus_di"]) > float(r["plus_di"])):
            out.append("جهت DI ناسازگار")
    elif reason in ("no_pullback", "no_trigger"):
        lo = i - C.PULLBACK_MAX
        seg = df.iloc[lo:i]
        mask = (seg["Low"] <= seg["ema20"]) if buy else (seg["High"] >= seg["ema20"])
        if not mask.any():
            out.append("لمس EMA20 در پنجره‌ی پولبک نبود")
        else:
            tpos = lo + int(np.argmax(mask.values))
            zone = df.iloc[tpos:i]
            bad = (zone["Close"] < zone["ema50"]).any() if buy else (zone["Close"] > zone["ema50"]).any()
            if reason == "no_pullback":
                if bad:
                    out.append("بسته شدن کندل بیرون از EMA50 در پولبک")
            else:
                rng = float(r["High"] - r["Low"])
                if not (c > float(zone["High"].max()) if buy else c < float(zone["Low"].min())):
                    out.append("close سقف/کف پولبک را نشکسته")
                if not (c > o if buy else c < o):
                    out.append("رنگ کندل خلاف جهت")
                if not (c > float(r["ema20"]) if buy else c < float(r["ema20"])):
                    out.append("close سمت اشتباهِ EMA20")
                if not (rng > 0 and abs(c - o) >= C.BODY_MIN * rng):
                    out.append("بدنه‌ی کندل کوچک")
    return out

def r_over_atr(df, i, side):
    """R/ATR the setup would have had (re-evaluated with the SL cap lifted)."""
    old = C.R_MAX_ATR
    C.R_MAX_ATR = 1e9
    try:
        res = S.evaluate_at(df, i, {}, None)
    finally:
        C.R_MAX_ATR = old
    return (res["r"] / res["atr"]) if res["status"] == "SIGNAL" else None

# ---------------------------------------------------------------- funnel over all candles
def analyze(df):
    n = len(df)
    first = {}
    causes = {k: {} for k in ("weak_trend", "low_adx", "no_pullback", "no_trigger")}
    wide, signals, inconsistent = [], 0, 0
    evaluated = 0
    for i in range(C.MIN_CANDLES - 1, n):
        res = S.evaluate_at(df, i, {}, None)
        evaluated += 1
        key = "SIGNAL" if res["status"] == "SIGNAL" else res["reason"]
        first[key] = first.get(key, 0) + 1
        if key == "SIGNAL":
            signals += 1
        elif key in causes:
            sc = sub_causes(df, i, key, res["side"])
            if not sc:
                inconsistent += 1
            for s in sc:
                causes[key][s] = causes[key].get(s, 0) + 1
        elif key == "sl_too_wide":
            v = r_over_atr(df, i, res["side"])
            if v is not None:
                wide.append(v)
    return {"evaluated": evaluated, "first": first, "causes": causes, "wide": wide, "signal_candles": signals,
            "inconsistent": inconsistent}

# ---------------------------------------------------------------- state-aware signal counter (no outcomes are reported)
def count_signals(df, spread=None):
    """Same lock logic as backtest.simulate (open signal, cooldown, new pullback after a signal). Returns counts only."""
    spread = C.BACKTEST_SPREAD_USD if spread is None else spread
    n, idx = len(df), df.index
    O, H, L, Cl = (df[c].values for c in ("Open", "High", "Low", "Close"))
    pos = {s: None for s in SIDES}
    last_sig = {s: None for s in SIDES}
    last_sl = {s: None for s in SIDES}
    signals = 0
    for i in range(C.MIN_CANDLES - 1, n):
        c_open = idx[i]; c_close = c_open + TF
        for side in SIDES:
            t = pos[side]
            if t is None:
                continue
            if t["state"] == "pending":
                entry = O[i] + spread if side == "BUY" else O[i]
                R = entry - t["sl"] if side == "BUY" else t["sl"] - entry
                if R <= 0:
                    pos[side] = None
                    continue
                t.update(state="open", entry=entry, R=R,
                         tp=entry + C.RR * R if side == "BUY" else entry - C.RR * R)
            mc = E.market_close_after(t["s_close"])
            if c_open >= mc:
                pos[side] = None
                continue
            if side == "BUY":
                sl_hit, tp_hit = L[i] <= t["sl"], H[i] >= t["tp"]
            else:
                sl_hit, tp_hit = H[i] + spread >= t["sl"], L[i] + spread <= t["tp"]
            if sl_hit:
                last_sl[side] = c_close; pos[side] = None
            elif tp_hit or c_close >= mc or (i - t["s_idx"]) >= C.EXPIRY_CANDLES:
                pos[side] = None
        ctx = {"open": {s: pos[s] is not None for s in SIDES},
               "last_signal_time": dict(last_sig), "last_sl_time": dict(last_sl)}
        r = S.evaluate_at(df, i, ctx, None)
        if r["status"] != "SIGNAL" or i >= n - 1:
            continue
        signals += 1
        last_sig[r["side"]] = c_close
        if idx[i + 1] == c_close:
            pos[r["side"]] = {"state": "pending", "sl": r["sl"], "s_idx": i, "s_close": c_close}
    return signals

VARIANTS = [
    ("فعلی (بدون تغییر)", {}),
    ("ADX_MIN  20 -> 15", {"ADX_MIN": 15}),
    ("ADX_MIN  20 -> 10", {"ADX_MIN": 10}),
    ("BODY_MIN  0.40 -> 0.30", {"BODY_MIN": 0.30}),
    ("BODY_MIN  0.40 -> 0.20", {"BODY_MIN": 0.20}),
    ("PULLBACK_MAX  5 -> 8", {"PULLBACK_MAX": 8}),
    ("PULLBACK_MAX  5 -> 12", {"PULLBACK_MAX": 12}),
    ("R_MAX_ATR  2.0 -> 2.5", {"R_MAX_ATR": 2.5}),
    ("R_MAX_ATR  2.0 -> 3.0", {"R_MAX_ATR": 3.0}),
    ("SLOPE_LOOKBACK  6 -> 3", {"SLOPE_LOOKBACK": 3}),
    ("RANGE_MAX_ATR  2.5 -> 3.5", {"RANGE_MAX_ATR": 3.5}),
    ("COOLDOWN_CANDLES  8 -> 0", {"COOLDOWN_CANDLES": 0}),
]

def sensitivity(df):
    out = []
    for label, change in VARIANTS:
        old = {k: getattr(C, k) for k in change}
        try:
            for k, v in change.items():
                setattr(C, k, v)
            out.append((label, count_signals(df)))
        finally:
            for k, v in old.items():
                setattr(C, k, v)
    return out

# ---------------------------------------------------------------- report
def build_report(df, info, sens):
    t0, t1 = df.index[C.MIN_CANDLES - 1], df.index[-1] + TF
    weeks = max((t1 - t0).total_seconds() / 86400 / 7, 1e-9)
    first = info["first"]
    in_win = info["evaluated"] - first.get("no_data", 0) - first.get("no_session", 0)
    L = ["=== DIAGNOSIS (counts only, no results) ===",
         f"data: {len(df)} candles M15, {C.SOURCE_TYPE}, evaluated {info['evaluated']} candles",
         f"period: {t0.isoformat()} -> {t1.isoformat()}  ({weeks:.1f} weeks)",
         f"candles inside the working window (05-23 Tehran, weekdays): {in_win}", "",
         "--- 1) first failing rule (share of in-window candles) ---"]
    for k in FUNNEL:
        if k in ("no_data", "no_session"):
            continue
        v = first.get(k, 0)
        L.append(f"{k:17s} {NAMES[k]:24s} {v:6d}  {100 * v / in_win if in_win else 0:5.1f}%")
    v = first.get("SIGNAL", 0)
    L.append(f"{'SIGNAL':17s} {'کندل سیگنال (بدون قفل)':24s} {v:6d}  {100 * v / in_win if in_win else 0:5.1f}%")
    L.append(f"(no_data={first.get('no_data', 0)}, outside window={first.get('no_session', 0)})")
    for key, title in (("weak_trend", "روند"), ("low_adx", "ADX / DI"), ("no_pullback", "پولبک"), ("no_trigger", "تریگر")):
        if info["causes"][key]:
            L += ["", f"--- details: {key} ({title}); one candle can fail for several reasons ---"]
            tot = first.get(key, 0)
            for s, c in sorted(info["causes"][key].items(), key=lambda x: -x[1]):
                L.append(f"{c:6d}  {100 * c / tot if tot else 0:5.1f}%  {s}")
    if info["wide"]:
        w = np.array(info["wide"])
        L += ["", f"--- sl_too_wide: R/ATR of the rejected setups (limit {C.R_MAX_ATR}) ---",
              f"count {len(w)}   min {w.min():.2f}   median {np.median(w):.2f}   90% {np.percentile(w, 90):.2f}   max {w.max():.2f}"]
    L += ["", "--- 2) how many signals per week (state-aware: locks, cooldown, new pullback) ---"]
    for label, n in sens:
        L.append(f"{label:30s} {n:4d} signals   {n / weeks:5.2f} per week")
    L += ["", "NOTE: counts only. Win rate / profit are NOT shown on purpose: this sample is far too small to judge a setup,",
          "and choosing a setting by looking at its results would only fit the past. More signals does not mean more profit.",
          f"consistency check (candles whose failure could not be explained): {info['inconsistent']}"]
    return "\n".join(L)

def main():
    raw = None
    for size in (5000, 1000):
        C.TD_OUTPUTSIZE = size
        try:
            raw = S.fetch_raw()
            break
        except RuntimeError as e:
            print("fetch with", size, "candles failed:", e)
    if raw is None:
        print("no data - stop")
        return 1
    df = S.add_indicators(clean_history(raw))
    if len(df) <= C.MIN_CANDLES:
        print(f"not enough candles: {len(df)}")
        return 1
    text = build_report(df, analyze(df), sensitivity(df))
    print(text)
    with open("diagnosis.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
