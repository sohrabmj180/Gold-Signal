"""
joint.py - READ ONLY. How do setups A, B and C behave TOGETHER on multi-year history?  Sends nothing, changes nothing.
Run from GitHub Actions (workflow "Joint"): needs the secret TD_KEY. Prints the report and writes joint_report.txt.
  1) each setup ALONE (own locks)        -> how many signals, how much do they overlap / contradict each other
  2) all three TOGETHER (shared locks)   -> what the live bot would actually have taken
Same conventions as backtest.py: signal on the close of candle S, entry at the NEXT open, BUY pays Ask, high spread model,
SL wins a same-candle double touch. Use it to UNDERSTAND the system, not to tune parameters (that would only fit the past).
"""
import os, sys, time
import numpy as np
import pandas as pd
import config as C
import strategy as S
import engine as E
import fib as F
import smc as M

TF = S.TF
SIDES = ("BUY", "SELL")
NEAR = 4                                   # "at the same time" = within 4 candles (1 hour)

# ---------------------------------------------------------------- long history from Twelve Data (paged backwards)
def _request(get, params, key, sleep):
    import requests
    for attempt in range(2):
        try:
            r = get("https://api.twelvedata.com/time_series", params=params, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError("TwelveData request failed: " + type(e).__name__) from None
        if r.status_code != 200:
            raise RuntimeError(f"TwelveData HTTP {r.status_code}")
        try:
            payload = r.json()
        except ValueError:
            raise RuntimeError("TwelveData returned non-JSON") from None
        if isinstance(payload, dict) and payload.get("code") == 429 and attempt == 0:
            print("rate limit reached - waiting 65 s"); sleep(65)       # free plan: 8 requests per minute
            continue
        return payload

def fetch_history(pages=8, sleep=time.sleep, get=None):
    """Latest `pages` x 5000 M15 candles (UTC), merged oldest -> newest. Stops quietly when the service has no more."""
    import requests
    get = get or requests.get
    key = os.getenv("TD_KEY")
    if not key:
        raise RuntimeError("TD_KEY is not set")
    frames, end = [], None
    for p in range(pages):
        params = {"symbol": getattr(C, "TD_SYMBOL", "XAU/USD"), "interval": "15min", "outputsize": 5000,
                  "timezone": "UTC", "apikey": key}
        if end:
            params["end_date"] = end
        payload = _request(get, params, key, sleep)
        try:
            df = S.parse_twelvedata(payload)
        except RuntimeError as e:
            print(f"page {p + 1}: stop ({S._scrub(e, key)})")
            break
        frames.append(df)
        print(f"page {p + 1}: {len(df)} candles  {df.index[0]} -> {df.index[-1]}")
        if len(df) < 200:
            break
        end = (df.index[0] - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        if p < pages - 1:
            sleep(8)                                     # free plan: at most 8 requests per minute
    if not frames:
        raise RuntimeError("no history received")
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]

def clean_history(raw, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if getattr(raw.index, "tz", None) is None:
        raise ValueError("source timezone unknown")
    df = raw[["Open", "High", "Low", "Close"]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert("UTC")
    return df[df.index + TF <= now]

# ---------------------------------------------------------------- candle-by-candle simulation (several setups)
def simulate_multi(df, evaluators, spread=None):
    """evaluators = [(name, fn)] tried in this order on every candle, exactly like the live engine: shared lock per side,
    one setup-C signal per Tehran day. Returns (closed_trades, info)."""
    spread = C.BACKTEST_SPREAD_USD if spread is None else spread
    n, idx = len(df), df.index
    O, H, L, Cl = (df[c].values for c in ("Open", "High", "Low", "Close"))
    pos = {s: None for s in SIDES}
    last_sig = {s: None for s in SIDES}
    last_sl = {s: None for s in SIDES}
    c_day = ""
    trades, info = [], {"signals": {}, "skipped_gap": 0, "skipped_entry_through_sl": 0, "open_at_end": 0}

    def close_trade(t, side, status, when, result):
        trades.append({"setup": t["setup"], "side": side, "signal_time": t["s_close"], "entry": t["entry"], "sl": t["sl"],
                       "tp": t["tp"], "R": t["R"], "status": status, "result_r": result, "close_time": when})
        pos[side] = None

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
                    info["skipped_entry_through_sl"] += 1
                    pos[side] = None
                    continue
                t.update(state="open", entry=entry, R=R, tp=entry + C.RR * R if side == "BUY" else entry - C.RR * R)
            mc = E.market_close_after(t["s_close"])
            if c_open >= mc:
                x = Cl[i - 1]
                res = (x - t["entry"]) / t["R"] if side == "BUY" else (t["entry"] - (x + spread)) / t["R"]
                close_trade(t, side, "EXPIRED", mc, res)
                continue
            if side == "BUY":
                sl_hit, tp_hit = L[i] <= t["sl"], H[i] >= t["tp"]
            else:
                sl_hit, tp_hit = H[i] + spread >= t["sl"], L[i] + spread <= t["tp"]
            if sl_hit:
                close_trade(t, side, "SL", c_close, -1.0)
                last_sl[side] = c_close
            elif tp_hit:
                close_trade(t, side, "TP", c_close, float(C.RR))
            elif c_close >= mc or (i - t["s_idx"]) >= C.EXPIRY_CANDLES:
                x = Cl[i]
                res = (x - t["entry"]) / t["R"] if side == "BUY" else (t["entry"] - (x + spread)) / t["R"]
                close_trade(t, side, "EXPIRED", c_close, res)
        for name, fn in evaluators:
            ctx = {"open": {s: pos[s] is not None for s in SIDES}, "last_signal_time": dict(last_sig),
                   "last_sl_time": dict(last_sl), "c_day": c_day}
            r = fn(df, i, ctx, None)
            if r["status"] != "SIGNAL" or i >= n - 1:
                continue
            info["signals"][name] = info["signals"].get(name, 0) + 1
            side = r["side"]
            last_sig[side] = c_close
            if name == "C":
                c_day = c_close.tz_convert("Asia/Tehran").date().isoformat()
            if idx[i + 1] != c_close:
                info["skipped_gap"] += 1
                continue
            pos[side] = {"state": "pending", "sl": r["sl"], "s_idx": i, "s_close": c_close, "setup": name}
    info["open_at_end"] = sum(1 for s in SIDES if pos[s] is not None)
    return trades, info

def evaluators(names=("A", "B", "C")):
    table = {"A": S.evaluate_at, "B": F.evaluate_fib_at, "C": M.evaluate_smc_at}
    return [(n, table[n]) for n in names]

# ---------------------------------------------------------------- statistics (same method as backtest.py)
def bootstrap_ci(r, n_boot=None, seed=None, batch=500):
    n_boot = n_boot or C.BOOTSTRAP_N
    rng = np.random.default_rng(C.BOOTSTRAP_SEED if seed is None else seed)
    means, done = [], 0
    while done < n_boot:
        k = min(batch, n_boot - done)
        means.append(r[rng.integers(0, len(r), size=(k, len(r)))].mean(axis=1))
        done += k
    m = np.concatenate(means)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def max_drawdown(r):
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float((peak - cum).max()) if len(r) else 0.0

def stats(trs, weeks):
    r = np.array([t["result_r"] for t in trs], float)
    n = len(r)
    out = {"n": n, "per_week": n / weeks if weeks > 0 else None, "win_rate": None, "expectancy_r": None,
           "ci_low": None, "ci_high": None, "profit_factor": None, "max_drawdown_r": None,
           "tp": sum(1 for t in trs if t["status"] == "TP"), "sl": sum(1 for t in trs if t["status"] == "SL"),
           "expired": sum(1 for t in trs if t["status"] == "EXPIRED")}
    if n == 0:
        return out
    out["win_rate"] = out["tp"] / n
    out["expectancy_r"] = float(r.mean())
    out["ci_low"], out["ci_high"] = bootstrap_ci(r)
    loss = -r[r < 0].sum()
    out["profit_factor"] = float(r[r > 0].sum() / loss) if loss > 0 else None
    out["max_drawdown_r"] = max_drawdown(r[np.argsort([t["close_time"] for t in trs], kind="stable")])
    return out

# ---------------------------------------------------------------- overlap and contradiction
def pair_overlap(xs, ys, near=NEAR):
    """How many trades of x have a trade of y at (about) the same time: same side = agree, opposite side = contradict."""
    win = pd.Timedelta(minutes=C.TIMEFRAME_MIN * near)
    agree = oppose = 0
    for a in xs:
        close = [b for b in ys if abs(b["signal_time"] - a["signal_time"]) <= win]
        agree += any(b["side"] == a["side"] for b in close)
        oppose += any(b["side"] != a["side"] for b in close)
    return {"x": len(xs), "agree": agree, "oppose": oppose}

def hedged_entries(xs, ys):
    """Trades of x opened while a trade of y in the OPPOSITE direction was still open (you would hold both directions)."""
    n = 0
    for a in xs:
        if any(b["side"] != a["side"] and b["signal_time"] <= a["signal_time"] < b["close_time"] for b in ys):
            n += 1
    return n

def confluence_split(trs_by_setup, near=NEAR):
    """For every setup: its trades split by 'another setup took the same direction at about the same time' or not."""
    win = pd.Timedelta(minutes=C.TIMEFRAME_MIN * near)
    out = {}
    for name, trs in trs_by_setup.items():
        others = [b for k, v in trs_by_setup.items() if k != name for b in v]
        yes, no = [], []
        for a in trs:
            partner = any(b["side"] == a["side"] and abs(b["signal_time"] - a["signal_time"]) <= win for b in others)
            (yes if partner else no).append(a)
        out[name] = (yes, no)
    return out

# ---------------------------------------------------------------- report
def _fmt(x, d=2, suffix=""):
    return "-" if x is None else f"{x:.{d}f}{suffix}"

def block(title, s):
    note = "" if s["n"] >= C.BACKTEST_MIN_TRADES else f"   [n < {C.BACKTEST_MIN_TRADES}: description only, no verdict]"
    return (f"{title}\n  closed trades: {s['n']}  (TP {s['tp']}, SL {s['sl']}, expired {s['expired']})   "
            f"per week: {_fmt(s['per_week'], 2)}\n"
            f"  win rate: {_fmt(None if s['win_rate'] is None else s['win_rate'] * 100, 1, '%')}   expectancy: "
            f"{_fmt(s['expectancy_r'], 3, ' R')}  95% CI [{_fmt(s['ci_low'], 3)} , {_fmt(s['ci_high'], 3)}]\n"
            f"  profit factor: {_fmt(s['profit_factor'])}   max drawdown: {_fmt(s['max_drawdown_r'], 2, ' R')}{note}\n")

def build_report(df, alone, alone_info, joint, joint_info):
    t0, t1 = df.index[C.MIN_CANDLES - 1], df.index[-1] + TF
    weeks = max((t1 - t0).total_seconds() / 86400 / 7, 1e-9)
    L = ["=== JOINT STUDY of setups A, B, C (read only) ===",
         f"data: {len(df)} candles M15, {C.SOURCE_TYPE}, price model {C.PRICE_MODEL}, spread {C.BACKTEST_SPREAD_USD} USD",
         f"period: {t0.isoformat()} -> {t1.isoformat()}  ({weeks:.1f} weeks = {weeks / 52.18:.2f} years)",
         "entry at the next open; BUY pays Ask; SL wins a same-candle double touch; news filter removed; window as in config", "",
         "--- 1) each setup ALONE ---"]
    for name in ("A", "B", "C"):
        L.append(block(f"setup {name}  (signals created: {alone_info[name]['signals'].get(name, 0)})", stats(alone[name], weeks)))
    L.append("--- 2) overlap / contradiction between setups (signals within 1 hour of each other) ---")
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        a, b = pair_overlap(alone[x], alone[y]), pair_overlap(alone[y], alone[x])
        L.append(f"{x} vs {y}:  of {a['x']} {x}-trades: {a['agree']} same direction, {a['oppose']} opposite  |  "
                 f"of {b['x']} {y}-trades: {b['agree']} same direction, {b['oppose']} opposite")
        L.append(f"          {x}-trades opened while a {y}-trade of the opposite direction was open: "
                 f"{hedged_entries(alone[x], alone[y])};   {y} while {x}: {hedged_entries(alone[y], alone[x])}")
    L += ["", "--- 3) 'confluence': a setup's trades split by whether another setup took the same direction at the same time ---"]
    for name, (yes, no) in confluence_split(alone).items():
        sy, sn = stats(yes, weeks), stats(no, weeks)
        L.append(f"{name}: with a partner n={sy['n']} (win rate {_fmt(None if sy['win_rate'] is None else sy['win_rate'] * 100, 1, '%')}, "
                 f"expectancy {_fmt(sy['expectancy_r'], 3, ' R')})   |   alone n={sn['n']} "
                 f"(win rate {_fmt(None if sn['win_rate'] is None else sn['win_rate'] * 100, 1, '%')}, "
                 f"expectancy {_fmt(sn['expectancy_r'], 3, ' R')})")
    L += ["", "--- 4) all three TOGETHER with shared locks (what the live bot would have taken) ---"]
    L.append(block("together", stats(joint, weeks)))
    for name in ("A", "B", "C"):
        part = [t for t in joint if t["setup"] == name]
        L.append(f"  from setup {name}: {len(part)} trades  (alone it would have had {len(alone[name])})")
    blocked = sum(len(v) for v in alone.values()) - len(joint)
    L.append(f"  signals lost to the shared lock: about {blocked}")
    L += ["", f"skipped: gap after signal {joint_info['skipped_gap']}, entry beyond SL {joint_info['skipped_entry_through_sl']}, "
              f"still open at the end (not counted) {joint_info['open_at_end']}",
          "", "NOTE: this describes how the setups relate to each other. It is NOT a reason to change any parameter:",
          "choosing numbers by looking at these results would only fit the past. No verdict is given for samples under "
          f"{C.BACKTEST_MIN_TRADES} trades."]
    return "\n".join(L)

def main():
    pages = int(os.getenv("JOINT_PAGES", "8"))
    raw = fetch_history(pages)
    df = S.add_indicators(clean_history(raw))
    if len(df) <= C.MIN_CANDLES:
        print(f"not enough candles: {len(df)}")
        return 1
    alone, alone_info = {}, {}
    for name, fn in evaluators():
        print("simulating setup", name, "alone ...", flush=True)
        alone[name], alone_info[name] = simulate_multi(df, [(name, fn)])
    print("simulating all three together ...", flush=True)
    joint, joint_info = simulate_multi(df, evaluators())
    text = build_report(df, alone, alone_info, joint, joint_info)
    print(text)
    with open("joint_report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
