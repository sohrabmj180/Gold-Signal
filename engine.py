"""Step 3: state, cooldown bookkeeping, expiry, scanning, late delivery.
Pure logic: Telegram is injected as send_fn(signal_dict, delayed_minutes_or_None) -> bool."""
import json
import os
import numpy as np
import pandas as pd
import config as C
import strategy as S

TF = S.TF
SIDES = ("BUY", "SELL")
_WD = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

# ---------------- state file ----------------
def default_state():
    return {"version": 1, "last_processed_time": None,
            "open": {"BUY": None, "SELL": None},              # id of the open signal per direction
            "last_signal_time": {"BUY": None, "SELL": None},  # close time of the last signal per direction
            "last_sl_time": {"BUY": None, "SELL": None},      # time the last SL was hit per direction
            "signals": {},                                    # id -> record (open or waiting for delivery)
            "unsent": [],                                     # ids not yet delivered to Telegram
            "reports": {"daily": "", "weekly": "", "monthly": ""},   # date of the last report sent
            "next_id": 1}

def load_state(path):
    """Missing file -> fresh state. Corrupt file -> raises (caller keeps old files and writes error.json)."""
    if not os.path.exists(path):
        return default_state()
    with open(path) as f:
        st = json.load(f)
    base = default_state()
    base.update({k: v for k, v in st.items() if k in base})   # unknown (old v1) keys are dropped
    return base

def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)                                     # atomic

def _ts(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def _ctx(state):
    return {"open": {s: state["open"][s] is not None for s in SIDES},
            "last_signal_time": {s: _ts(state["last_signal_time"][s]) for s in SIDES},
            "last_sl_time": {s: _ts(state["last_sl_time"][s]) for s in SIDES}}

# ---------------- closing rules ----------------
def market_close_after(ts):
    """First weekly market close (Friday 21:00 UTC) strictly after ts."""
    day, hhmm = C.MARKET_CLOSE_UTC
    h, m = hhmm.split(":")
    base = ts.normalize() + pd.Timedelta(hours=int(h), minutes=int(m))
    cand = base + pd.Timedelta(days=(_WD[day] - ts.weekday()) % 7)
    return cand + pd.Timedelta(days=7) if cand <= ts else cand

def expiry_result(sig, exit_close):
    """R result at expiry, marked to the last close. BUY exits on Bid, SELL exits on Ask (+spread)."""
    R = sig["r"]
    if sig["side"] == "BUY":
        return round((exit_close - sig["entry"]) / R, 2)
    return round((sig["entry"] - (exit_close + C.SPREAD_USD)) / R, 2)

def _close(rec, status, when, result_r):
    rec["status"] = status
    rec["close_time_utc"] = when.isoformat()
    rec["result_r"] = result_r

def _update_open(df, p, state, events):
    for side in SIDES:
        sid = state["open"][side]
        if sid is None:
            continue
        rec = state["signals"][str(sid)]
        s_close = _ts(rec["time_utc"])
        c_open = df.index[p]
        c_close = c_open + TF
        if c_open < s_close:                                  # candle is not after the signal candle
            continue
        mc = market_close_after(s_close)
        if c_open >= mc:                                      # market already closed before this candle
            _close(rec, "EXPIRED", mc, expiry_result(rec, float(df["Close"].iloc[p - 1])))
        else:
            hit = S.check_hit(rec, df.iloc[p])
            if hit == "SL":
                _close(rec, "SL", c_close, -1.0)
                state["last_sl_time"][side] = c_close.isoformat()
            elif hit == "TP":
                _close(rec, "TP", c_close, float(C.RR))
            else:
                s_open = s_close - TF
                k = int(((df.index > s_open) & (df.index <= c_open)).sum())   # real candles after S
                if c_close >= mc or k >= C.EXPIRY_CANDLES:
                    _close(rec, "EXPIRED", c_close, expiry_result(rec, float(df["Close"].iloc[p])))
        if rec["status"] != "OPEN":
            state["open"][side] = None
            events.append({"type": "SIGNAL_CLOSED", "signal": dict(rec)})

def _register(res, state, events):
    sid = state["next_id"]
    state["next_id"] = sid + 1
    rec = dict(res)
    rec.update(id=sid, status="OPEN", sent=False, delivery="pending", mode=S.effective_mode(),
               close_time_utc=None, result_r=None, delayed_minutes=None)
    state["signals"][str(sid)] = rec
    state["open"][rec["side"]] = sid
    state["last_signal_time"][rec["side"]] = rec["time_utc"]
    state["unsent"].append(sid)
    events.append({"type": "SIGNAL_CREATED", "signal": dict(rec)})

# ---------------- delivery ----------------
def _deliver(df, state, now, events, send_fn, sleep_fn):
    last = df.iloc[-1]
    last_close_ts = df.index[-1] + TF
    atr, c = float(last["atr"]), float(last["Close"])
    for sid in list(state["unsent"]):
        rec = state["signals"].get(str(sid))
        if rec is None:
            state["unsent"].remove(sid)
            continue
        if rec["status"] != "OPEN":                           # SL/TP/expiry already happened
            rec["delivery"] = "skipped_closed"
            state["unsent"].remove(sid)
            events.append({"type": "SKIPPED", "why": "already_closed", "signal": dict(rec)})
            continue
        if abs(c - rec["reference_price"]) > C.LATE_MAX_ATR * atr:
            rec["delivery"] = "skipped_late"
            state["unsent"].remove(sid)
            events.append({"type": "SKIPPED", "why": "price_moved", "signal": dict(rec)})
            continue
        s_close = _ts(rec["time_utc"])
        late = s_close < last_close_ts
        delay = max(0, int(round((now - s_close) / pd.Timedelta(minutes=1))))
        rec["delayed_minutes"] = delay
        rec["late"] = bool(late)
        ok = False
        for attempt in range(C.TG_RETRIES):
            res = send_fn(dict(rec), delay if late else None) if send_fn else False
            ok = bool(res)
            if ok:
                if isinstance(res, int) and not isinstance(res, bool):
                    rec["msg_id"] = res                    # Telegram message id (for reply on TP/SL)
                break
            if sleep_fn and attempt < C.TG_RETRIES - 1:
                sleep_fn(2)
        if ok:
            rec["sent"] = True
            rec["delivery"] = "sent"
            state["unsent"].remove(sid)
            events.append({"type": "DELIVERED", "signal": dict(rec), "delayed_minutes": delay if late else None})
        else:
            events.append({"type": "SEND_FAILED", "signal": dict(rec)})      # stays pending, retried next run

# ---------------- main entry ----------------
def process(df, state, news=None, now_utc=None, send_fn=None, sleep_fn=None):
    """df: closed candles with indicators (from strategy.prepare, ok=True). Mutates `state` in place.
    Returns {"events", "last_eval", "scanned", "skipped_old"}."""
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _ts(now_utc)
    events = []
    lp = _ts(state["last_processed_time"])
    if lp is None:
        pos = [len(df) - 1]                                   # first run: only the newest closed candle
    else:
        pos = [int(x) for x in np.flatnonzero(df.index > lp)]
    new_pos = pos[-C.MAX_SCAN_CANDLES:]
    old_pos = pos[:-C.MAX_SCAN_CANDLES] if len(pos) > C.MAX_SCAN_CANDLES else []
    for p in old_pos:                                         # too old to signal on, but open signals must still be tracked
        _update_open(df, p, state, events)
    last_eval = None
    for p in new_pos:
        _update_open(df, p, state, events)
        res = S.evaluate_at(df, p, _ctx(state), news)
        last_eval = res
        if res["status"] == "SIGNAL":
            _register(res, state, events)
    if new_pos:
        state["last_processed_time"] = df.index[-1].isoformat()
    else:
        last_eval = S.evaluate_at(df, len(df) - 1, _ctx(state), news)      # read-only, for the website reason
    _deliver(df, state, now, events, send_fn, sleep_fn)
    for sid, rec in list(state["signals"].items()):           # forget finished + already handled signals
        if rec["status"] != "OPEN" and rec["delivery"] != "pending":
            del state["signals"][sid]
    return {"events": events, "last_eval": last_eval, "scanned": len(new_pos), "skipped_old": len(old_pos)}
# =====================================================================
# Two setups: A (EMA pullback, strategy.py) and B (Fibonacci pullback, fib.py).
# Defined last on purpose: these replace the earlier _register / process above.
# Both setups share the same locks per direction (one open signal per side) and the same state file.
# =====================================================================
import fib as _fib

def _register(res, state, events):
    sid = state["next_id"]
    state["next_id"] = sid + 1
    rec = dict(res)
    rec.setdefault("setup", "A")
    rec.update(id=sid, status="OPEN", sent=False, delivery="pending", mode=S.effective_mode(),
               close_time_utc=None, result_r=None, delayed_minutes=None)
    state["signals"][str(sid)] = rec
    state["open"][rec["side"]] = sid
    state["last_signal_time"][rec["side"]] = rec["time_utc"]
    state["unsent"].append(sid)
    events.append({"type": "SIGNAL_CREATED", "signal": dict(rec)})

def process(df, state, news=None, now_utc=None, send_fn=None, sleep_fn=None):
    """df: closed candles with indicators (from strategy.prepare, ok=True). Mutates `state` in place.
    Every new candle is evaluated by setup A, then by setup B (if enabled)."""
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _ts(now_utc)
    events = []
    lp = _ts(state["last_processed_time"])
    if lp is None:
        pos = [len(df) - 1]                                   # first run: only the newest closed candle
    else:
        pos = [int(x) for x in np.flatnonzero(df.index > lp)]
    new_pos = pos[-C.MAX_SCAN_CANDLES:]
    old_pos = pos[:-C.MAX_SCAN_CANDLES] if len(pos) > C.MAX_SCAN_CANDLES else []
    for p in old_pos:                                         # too old to signal on, but open signals must still be tracked
        _update_open(df, p, state, events)
    last_eval = None
    for p in new_pos:
        _update_open(df, p, state, events)
        res = S.evaluate_at(df, p, _ctx(state), news)
        res.setdefault("setup", "A")
        last_eval = res
        if res["status"] == "SIGNAL":
            _register(res, state, events)
        if getattr(C, "ENABLE_SETUP_B", True):
            rb = _fib.evaluate_fib_at(df, p, _ctx(state), news)
            if rb["status"] == "SIGNAL":
                _register(rb, state, events)
    if new_pos:
        state["last_processed_time"] = df.index[-1].isoformat()
    else:
        last_eval = S.evaluate_at(df, len(df) - 1, _ctx(state), news)      # read-only, for the website reason
    _deliver(df, state, now, events, send_fn, sleep_fn)
    for sid, rec in list(state["signals"].items()):           # forget finished + already handled signals
        if rec["status"] != "OPEN" and rec["delivery"] != "pending":
            del state["signals"][sid]
    return {"events": events, "last_eval": last_eval, "scanned": len(new_pos), "skipped_old": len(old_pos)}
# =====================================================================
# Three setups: A (EMA pullback), B (Fibonacci pullback), C (Smart Money: CHoCH + BOS + Order Block).
# Defined last on purpose: these replace the earlier default_state / _register / process above.
# All setups share the locks per direction. Setup C is limited to ONE signal per Tehran day (state key "c_day").
# =====================================================================
import smc as _smc

_default_state_prev = default_state
_register_prev = _register

def default_state():
    st = _default_state_prev()
    st["c_day"] = ""                                          # Tehran date of the last setup-C signal
    return st

def _register(res, state, events):
    _register_prev(res, state, events)
    if res.get("setup") == "C":
        state["c_day"] = pd.Timestamp(res["time_utc"]).tz_convert("Asia/Tehran").date().isoformat()

def _ctx_full(state):
    ctx = _ctx(state)
    ctx["c_day"] = state.get("c_day", "")
    return ctx

def process(df, state, news=None, now_utc=None, send_fn=None, sleep_fn=None):
    """df: closed candles with indicators (from strategy.prepare, ok=True). Mutates `state` in place.
    Every new candle is evaluated by setup A, then B and C (each can be switched off in config.py)."""
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _ts(now_utc)
    events = []
    lp = _ts(state["last_processed_time"])
    if lp is None:
        pos = [len(df) - 1]                                   # first run: only the newest closed candle
    else:
        pos = [int(x) for x in np.flatnonzero(df.index > lp)]
    new_pos = pos[-C.MAX_SCAN_CANDLES:]
    old_pos = pos[:-C.MAX_SCAN_CANDLES] if len(pos) > C.MAX_SCAN_CANDLES else []
    for p in old_pos:                                         # too old to signal on, but open signals must still be tracked
        _update_open(df, p, state, events)
    last_eval = None
    for p in new_pos:
        _update_open(df, p, state, events)
        res = S.evaluate_at(df, p, _ctx_full(state), news)
        res.setdefault("setup", "A")
        last_eval = res
        if res["status"] == "SIGNAL":
            _register(res, state, events)
        for flag, fn in (("ENABLE_SETUP_B", _fib.evaluate_fib_at), ("ENABLE_SETUP_C", _smc.evaluate_smc_at)):
            if getattr(C, flag, True):
                rx = fn(df, p, _ctx_full(state), news)
                if rx["status"] == "SIGNAL":
                    _register(rx, state, events)
    if new_pos:
        state["last_processed_time"] = df.index[-1].isoformat()
    else:
        last_eval = S.evaluate_at(df, len(df) - 1, _ctx_full(state), news)   # read-only, for the website reason
    _deliver(df, state, now, events, send_fn, sleep_fn)
    for sid, rec in list(state["signals"].items()):           # forget finished + already handled signals
        if rec["status"] != "OPEN" and rec["delivery"] != "pending":
            del state["signals"][sid]
    return {"events": events, "last_eval": last_eval, "scanned": len(new_pos), "skipped_old": len(old_pos)}
