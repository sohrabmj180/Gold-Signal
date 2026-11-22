"""
Gold signal bot - GitHub Actions entry point (runs once per call).
Env: TG_TOKEN, TG_CHAT_ID.   Rules live in strategy.py, state handling in engine.py, constants in config.py.
The bot NEVER edits config.py (MODE is changed by the owner by hand).
Test/educational signals only - not financial advice.
"""
import os, io, re, csv, json, math, time, tempfile, traceback
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import jdatetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import strategy as S
import engine as E

TZ = ZoneInfo("Asia/Tehran")
TF = S.TF

# ------------------------------------------------------------------ helpers
def paths(wd="."):
    j = lambda n: os.path.join(wd, n)
    return {"signal": j(C.SIGNAL_FILE), "log": j(C.LOG_FILE), "state": j(C.STATE_FILE),
            "error": j(C.ERROR_FILE), "news": j(C.NEWS_FILE),
            "chart": os.path.join(tempfile.gettempdir(), "gold_chart.png")}

def mask(text):
    """Never let the Telegram token reach a log or a public file."""
    text = str(text)
    tok = os.getenv("TG_TOKEN")
    if tok:
        text = text.replace(tok, "***")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot***", text)

def clean(o):
    """Make any structure JSON-safe: NaN / Infinity -> None."""
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o

def dumps(obj):
    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False, indent=1)

def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)

def write_if_changed(path, text, normalize=None):
    """Write atomically only when the content really changed (keeps git history quiet)."""
    try:
        with open(path, encoding="utf-8") as f:
            old = f.read()
        if old == text or (normalize and normalize(old) == normalize(text)):
            return False
    except FileNotFoundError:
        pass
    atomic_write(path, text)
    return True

def _without_generated_at(text):
    try:
        d = json.loads(text); d.pop("generated_at", None); return d
    except Exception:
        return text

# ------------------------------------------------------------------ telegram
def tg(method, data=None, files=None):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not (tok and chat):
        print("Telegram not configured")
        return None
    data = dict(data or {}); data["chat_id"] = chat
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/{method}", data=data, files=files, timeout=30)
        j = r.json()
        if not j.get("ok"):
            print("Telegram error:", mask(j))
            return None
        return j.get("result") or {}
    except Exception as e:
        print("Telegram exception:", mask(e))
        return None

_TAG = {"B": "Fib", "C": "SMC"}                            # label shown next to the symbol for setups B and C

def _mode_prefix():
    return "🧪 آزمایشی | " if S.effective_mode() == "TEST" else ""

def make_caption(sig, delay):
    head = f"{_mode_prefix()}{C.NAME} {C.SIDE_LABEL[sig['side']]}" + (f" | {_TAG[sig['setup']]}" if sig.get("setup") in _TAG else "")
    if C.SOURCE_TYPE == "FUTURES":                         # no absolute prices on futures data
        lines = [head, "Entry: ≈ market", f"SL distance: {sig['sl_distance']:.2f}$",
                 f"TP distance: {sig['tp_distance']:.2f}$"]
    else:
        lines = [head, f"Entry: ≈ {sig['entry']}", f"TP: {sig['tp']}", f"SL: {sig['sl']}"]
    if delay:
        lines.append(f"⏱ تأخیر {delay} دقیقه")
    lines.append(C.DISCLAIMER)
    return "\n".join(lines)

def make_chart(df, sig, path):
    """Candles + a TradingView-style Long/Short position tool: profit box (entry->TP) and loss box (entry->SL)."""
    from matplotlib.patches import Rectangle
    s_open = (pd.Timestamp(sig["time_utc"]) - TF) if sig.get("time_utc") else None
    s_pos = int(df.index.get_indexer([s_open])[0]) if s_open is not None else -1
    if s_pos < 0:
        s_pos = len(df) - 1
    start = max(0, s_pos - 45)
    d = df.iloc[start:]
    n, xs = len(d), s_pos - start
    futures = C.SOURCE_TYPE == "FUTURES"
    bg, fg, grid = "#0b1620", "#c9d1d9", "#2a3a46"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for x, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r["Close"] >= r["Open"] else "#ef5350"
        ax.vlines(x, r["Low"], r["High"], color=col, linewidth=1)
        ax.vlines(x, r["Open"], r["Close"], color=col, linewidth=4)
    entry, tp, sl = sig["entry"], sig["tp"], sig["sl"]
    x0, x1 = xs + 0.5, n - 1 + 14
    for a, b, colr in ((min(entry, tp), max(entry, tp), "#26a69a"), (min(entry, sl), max(entry, sl), "#ef5350")):
        ax.add_patch(Rectangle((x0, a), x1 - x0, b - a, facecolor=colr, alpha=0.22, edgecolor=colr, linewidth=1))
    ax.plot([x0, x1], [entry, entry], color="#4c8dff", linewidth=1.3)
    mid = (x0 + x1) / 2
    fib_lo = fib_hi = None
    if sig.get("setup") == "B" and "leg_start" in sig:           # Fibonacci retracement of the impulse leg
        la, lb = sig["leg_start"], sig["leg_end"]
        levels = [(lv, lb + (la - lb) * lv) for lv in (0.0, 0.382, 0.5, 0.618, 0.786)]
        for lv, y in levels:
            ax.axhline(y, color="#f0b90b", linestyle=":", linewidth=0.8, alpha=0.75)
            ax.text(-0.5, y, f"{lv * 100:.1f}%", color="#f0b90b", fontsize=6, va="bottom", ha="left")
        fib_lo, fib_hi = min(y for _, y in levels), max(y for _, y in levels)
    if sig.get("setup") == "C" and "ob_low" in sig:               # Smart Money: order block zone + BOS / CHoCH levels
        oi = int(d.index.get_indexer([pd.Timestamp(sig["ob_time_utc"])])[0])
        oi = max(oi, 0)
        ax.add_patch(Rectangle((oi - 0.4, sig["ob_low"]), x1 - (oi - 0.4), sig["ob_high"] - sig["ob_low"],
                               facecolor="#4c8dff", alpha=0.22, edgecolor="#4c8dff", linewidth=1))
        ax.text(oi - 0.4, sig["ob_high"], " OB", color="#4c8dff", fontsize=7, va="bottom", ha="left")
        for key, label in (("bos_level", "BOS"), ("choch_level", "CHoCH")):
            ax.axhline(sig[key], color="#f0b90b", linestyle=":", linewidth=0.8, alpha=0.75)
            ax.text(-0.5, sig[key], label, color="#f0b90b", fontsize=6, va="bottom", ha="left")
        ys = (sig["ob_low"], sig["ob_high"], sig["bos_level"], sig["choch_level"])
        fib_lo, fib_hi = min(ys), max(ys)
    tp_txt = f"TP  {abs(tp - entry):.1f}$  ({C.RR:g}R)" if futures else f"TP  {tp}"
    sl_txt = f"SL  {abs(sl - entry):.1f}$  (1R)" if futures else f"SL  {sl}"
    ax.text(mid, (entry + tp) / 2, tp_txt, color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(mid, (entry + sl) / 2, sl_txt, color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(x1 - 0.3, entry, "Entry ≈ " if futures else f"Entry ≈ {entry}", color="#4c8dff",
            ha="right", va="bottom", fontsize=7)
    lo = min(float(d["Low"].min()), sl, tp); hi = max(float(d["High"].max()), sl, tp)
    if fib_lo is not None:
        lo, hi = min(lo, fib_lo), max(hi, fib_hi)
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-1, x1 + 1)
    side_col = "#26a69a" if sig["side"] == "BUY" else "#ef5350"
    ax.set_title(f"{C.NAME} {C.SIDE_LABEL[sig['side']].upper()}  M15" + (("  " + _TAG[sig["setup"]]) if sig.get("setup") in _TAG else "")
                 + ("   TEST" if S.effective_mode() == "TEST" else ""),
                 fontsize=11, color=side_col, loc="left", fontweight="bold")
    ax.grid(color=grid, alpha=0.25, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.set_xticks([])
    if futures:
        ax.set_yticks([])                                  # hide absolute futures prices
    else:
        ax.yaxis.tick_right(); ax.tick_params(axis="y", colors=fg, labelsize=7)
    fig.tight_layout(); fig.savefig(path, facecolor=bg); plt.close(fig)
    return path

def make_sender(df, tg_fn, P):
    def send(sig, delay):
        with open(make_chart(df, sig, P["chart"]), "rb") as f:
            res = tg_fn("sendPhoto", {"caption": make_caption(sig, delay)}, {"photo": f})
        return res["message_id"] if res and "message_id" in res else False
    return send

def result_text(sig):
    if sig["status"] == "TP":
        return f"Tp ✅ {float(sig['result_r']):+g}R"
    if sig["status"] == "SL":
        return f"SL ❌ {float(sig['result_r']):+g}R"
    return f"Expired ⏱ {float(sig['result_r']):+g}R"

def send_reply(tg_fn, sig):
    for _ in range(C.TG_RETRIES):
        if tg_fn("sendMessage", {"text": result_text(sig), "reply_to_message_id": sig["msg_id"]}) is not None:
            return True
    return False

# ------------------------------------------------------------------ log (signals_log.csv)
def load_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def log_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=C.LOG_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in C.LOG_COLUMNS})
    return buf.getvalue()

def _row(sg):
    return {"id": sg["id"], "time_utc": sg["time_utc"], "side": sg["side"], "reference": sg["reference_price"],
            "entry": sg["entry"], "sl": sg["sl"], "tp": sg["tp"], "r": sg["r"], "status": sg["status"],
            "result_r": sg["result_r"], "close_time_utc": sg["close_time_utc"], "delivery": sg["delivery"],
            "setup": sg.get("setup", "A")}

def apply_events(rows, events):
    byid = {str(r["id"]): r for r in rows}
    for e in events:
        sg, key = e["signal"], str(e["signal"]["id"])
        if e["type"] == "SIGNAL_CREATED":
            row = _row(sg); byid[key] = row; rows.append(row)
        elif key in byid:
            byid[key].update({"status": sg["status"], "result_r": sg["result_r"],
                              "close_time_utc": sg["close_time_utc"], "delivery": sg["delivery"]})
    return rows

# ------------------------------------------------------------------ site payload
def build_payload(prep, res, state, news_stale, now):
    futures = C.SOURCE_TYPE == "FUTURES"
    p = {"status": "HOLD", "reason": None, "time_utc": None, "price": None, "entry": None, "sl": None, "tp": None,
         "sl_distance": None, "tp_distance": None, "r": None, "atr": None, "adx": None, "above_ema200": None,
         "delayed_minutes": None, "mode": S.effective_mode(), "source_type": C.SOURCE_TYPE,
         "news_file_stale": bool(news_stale), "disclaimer": C.DISCLAIMER, "generated_at": now.isoformat(), "setup": None}
    if not prep["ok"]:
        p["reason"] = prep["reason"]
        return p
    last = prep["df"].iloc[-1]
    p.update(atr=round(float(last["atr"]), 2), adx=round(float(last["adx"]), 2),
             above_ema200=bool(last["Close"] > last["ema200"]),
             price=None if futures else round(float(last["Close"]), 2))
    shown = [r for r in state["signals"].values() if r["status"] == "OPEN" and r["delivery"] == "sent"]
    if shown:                                              # only signals that were really published
        r = max(shown, key=lambda x: x["time_utc"])
        p.update(status=r["side"], reason=None, time_utc=r["time_utc"], setup=r.get("setup", "A"),
                 entry=None if futures else r["entry"], sl=None if futures else r["sl"],
                 tp=None if futures else r["tp"], sl_distance=r["sl_distance"], tp_distance=r["tp_distance"],
                 r=round(r["tp_distance"] / r["sl_distance"], 2),
                 delayed_minutes=int(r["delayed_minutes"]) if r.get("late") else 0)
        return p
    ev = (res or {}).get("last_eval") or {}
    p["reason"] = ev.get("reason") if ev.get("status") == "HOLD" else "open_signal"
    p["time_utc"] = ev.get("time_utc")
    return p

# ------------------------------------------------------------------ reports (from the log, published signals only)
def jal(d):
    j = jdatetime.date.fromgregorian(date=d)
    return j.year, j.month, j.day

def closed_published(rows):
    out = []
    for r in rows:
        if r.get("status") in ("TP", "SL", "EXPIRED") and r.get("delivery") == "sent" and r.get("close_time_utc"):
            out.append((pd.Timestamp(r["close_time_utc"]).tz_convert(TZ).date(), r))
    return out

def report_text(title, trs):
    label = "🧪 آزمایشی | " if S.effective_mode() == "TEST" else ""
    if not trs:
        return f"📊 {label}{title}\n\nسیگنال بسته‌شده‌ای نبود."
    n = len(trs)
    w = sum(1 for t in trs if t["status"] == "TP"); l = sum(1 for t in trs if t["status"] == "SL"); x = n - w - l
    net = sum(float(t["result_r"]) for t in trs)
    return (f"📊 {label}{title}\n\nتعداد سیگنال بسته‌شده: {n}\n✅ TP: {w}\n❌ SL: {l}\n⏱ انقضا: {x}\n"
            f"نرخ برد: {w * 100 / n:.0f}%\nمجموع: {net:+.2f}R\n{C.DISCLAIMER}")

def check_reports(rows, state, now, tg_fn):
    t = now.tz_convert(TZ)
    if t.hour < C.REPORT_HOUR:
        return False
    rep = state.setdefault("reports", {"daily": "", "weekly": "", "monthly": ""})
    today, iso, changed = t.date(), t.date().isoformat(), False
    closed = closed_published(rows)
    if rep["daily"] != iso:
        trs = [r for d, r in closed if d == today]
        if trs or C.REPORT_EMPTY_DAILY:
            ok = tg_fn("sendMessage", {"text": report_text("گزارش روزانه", trs)}) is not None
        else:
            ok = True
        if ok:
            rep["daily"] = iso; changed = True
    if t.weekday() == 4 and rep["weekly"] != iso:                       # Friday: Saturday..Friday
        start = today - dt.timedelta(days=6)
        trs = [r for d, r in closed if start <= d <= today]
        if tg_fn("sendMessage", {"text": report_text("گزارش هفتگی", trs)}) is not None:
            rep["weekly"] = iso; changed = True
    if jal(today + dt.timedelta(days=1))[2] == 1 and rep["monthly"] != iso:   # last day of the Jalali month
        ym = jal(today)[:2]
        trs = [r for d, r in closed if jal(d)[:2] == ym]
        if tg_fn("sendMessage", {"text": report_text("گزارش ماهانه", trs)}) is not None:
            rep["monthly"] = iso; changed = True
    return changed

# ------------------------------------------------------------------ one run
def run(now=None, fetch=None, tg_fn=None, workdir=".", sleep_fn=time.sleep):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    fetch, tg_fn, P = fetch or S.fetch_raw, tg_fn or tg, paths(workdir)

    state = E.load_state(P["state"])                       # corrupt file -> exception -> error.json, old files stay
    rows = load_log(P["log"])
    news = S.load_news(P["news"])
    stale = S.news_file_stale(news, now)
    prep = S.prepare(fetch(), now)

    res, events = None, []
    if prep["ok"]:
        res = E.process(prep["df"], state, news, now, make_sender(prep["df"], tg_fn, P), sleep_fn)
        events = res["events"]
        apply_events(rows, events)
    write_if_changed(P["log"], log_text(rows))             # save BEFORE replies so a crash cannot re-send signals
    write_if_changed(P["state"], dumps(state))
    for e in events:
        if e["type"] == "SIGNAL_CLOSED" and e["signal"].get("msg_id"):
            send_reply(tg_fn, e["signal"])
    if check_reports(rows, state, now, tg_fn):
        write_if_changed(P["state"], dumps(state))
    payload = build_payload(prep, res, state, stale, now)
    write_if_changed(P["signal"], dumps(payload), normalize=_without_generated_at)
    return payload

def main():
    P = paths(".")
    try:
        run()
        if os.path.exists(P["error"]):
            os.remove(P["error"])
    except Exception as e:                                 # keep the previous signal.json, record the error
        err = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": mask(repr(e)),
               "trace": mask(traceback.format_exc()[-1500:])}
        atomic_write(P["error"], dumps(err))
        print("ERROR:", err["error"])

if __name__ == "__main__":
    main()
