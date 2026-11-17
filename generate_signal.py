"""Gold signal bot (runs every few minutes via GitHub Actions).

1. Computes the signal -> signal.json (website)
2. Sends new signals to Telegram and records them in trades.json
3. Detects TP / SL hits automatically and replies to the original signal
4. Sends image reports to the channel:
     - daily   at 22:00 Iran time (every day with closed trades)
     - weekly  on Friday 22:00 Iran time
     - monthly on the last Friday of the month, 22:00 Iran time
"""
import json, datetime, os
from zoneinfo import ZoneInfo
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

SYMBOL, INTERVAL = "GC=F", "5m"
ATR_SL, ATR_TP = 1.5, 3.0

# ---- Message formats: edit freely ----
NAME = "XAUUSD"
TEMPLATE = "{name} {side}\nEntry: {entry}\nTP: {tp}\nSL: {sl}"
SIDE = {"BUY": "Buy", "SELL": "Sell"}
MSG_TP = "Tp ✅  {r}R"          # reply when take-profit is hit
MSG_SL = "Sl ❌  -1R"           # reply when stop-loss is hit
CAP_DAILY = "📊 Daily report · {date}\nTP: {tp} | SL: {sl}\nNet: {net}"
CAP_WEEKLY = "📅 Weekly report · {date}\nTP: {tp} | SL: {sl}\nNet: {net}"
CAP_MONTHLY = "🗓 Monthly report · {date}\nTP: {tp} | SL: {sl}\nNet: {net}"

TZ = ZoneInfo("Asia/Tehran")      # report schedule uses Iran time
REPORT_HOUR = 22                  # 22:00 Iran time
TRADES_FILE = "trades.json"

# Trades that were already traded by hand before the bot tracked them
# (used only the first time, when trades.json does not exist yet).
SEED = [
    {"id": "seed1", "side": "BUY", "entry": 3863.1, "tp": 3873.43, "sl": 3857.93,
     "open": "2026-11-17T00:56:00+00:00", "status": "sl",
     "close": "2026-11-17T04:05:00+00:00", "r": -1.0, "pts": -5.17, "msg_id": None},
    {"id": "seed2", "side": "BUY", "entry": 3868.9, "tp": 3886.77, "sl": 3859.97,
     "open": "2026-11-17T02:48:00+00:00", "status": "tp",
     "close": "2026-11-17T04:05:00+00:00", "r": 2.0, "pts": 17.87, "msg_id": None},
]


# --------------------------------------------------------------- telegram
def tg_ready():
    return bool(os.getenv("TG_TOKEN") and os.getenv("TG_CHAT_ID"))


def _tg(method, data, files=None):
    """Returns the Telegram message_id on success, otherwise None. Never raises."""
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not (tok and chat):
        print("Telegram not configured")
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/{method}",
                          data={"chat_id": chat, **data}, files=files, timeout=30)
        print("Telegram:", method, r.status_code, r.text[:100])
        if r.status_code == 200:
            return r.json()["result"]["message_id"]
    except Exception as e:  # network problem etc.
        print("Telegram error:", e)
    return None


def tg_photo(path, caption, reply_to=None):
    data = {"caption": caption}
    if reply_to:
        data["reply_to_message_id"] = reply_to
    with open(path, "rb") as f:
        return _tg("sendPhoto", data, files={"photo": f})


def tg_text(text, reply_to=None):
    data = {"text": text}
    if reply_to:
        data["reply_to_message_id"] = reply_to
    return _tg("sendMessage", data)


# ------------------------------------------------------------ trades file
def load_state():
    try:
        with open(TRADES_FILE) as f:
            st = json.load(f)
        st.setdefault("trades", [])
        st.setdefault("sent", {})
        return st
    except Exception:
        return {"trades": [dict(t) for t in SEED], "sent": {}}


def save_state(st):
    with open(TRADES_FILE, "w") as f:
        json.dump(st, f, indent=1)


def new_trade(out, ts, msg_id=None):
    return {"id": ts.isoformat(), "side": out["signal"], "entry": out["entry"],
            "tp": out["tp"], "sl": out["sl"], "open": ts.isoformat(),
            "status": "open", "close": None, "r": None, "pts": None, "msg_id": msg_id}


def update_trades(st, df, notify=None):
    """Looks at candles after each open trade; marks TP / SL. SL wins if one candle hits both."""
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    hi, lo = df["High"].values, df["Low"].values
    for t in st["trades"]:
        if t["status"] != "open":
            continue
        opened = pd.Timestamp(t["open"])
        for ts, h, l in zip(idx[idx > opened], hi[idx > opened], lo[idx > opened]):
            if t["side"] == "BUY":
                hit_sl, hit_tp = l <= t["sl"], h >= t["tp"]
            else:
                hit_sl, hit_tp = h >= t["sl"], l <= t["tp"]
            if not (hit_sl or hit_tp):
                continue
            win = hit_tp and not hit_sl
            sign = 1 if t["side"] == "BUY" else -1
            t["status"] = "tp" if win else "sl"
            t["close"] = ts.isoformat()
            t["r"] = round(abs(t["tp"] - t["entry"]) / abs(t["entry"] - t["sl"]), 2) if win else -1.0
            t["pts"] = round(sign * ((t["tp"] if win else t["sl"]) - t["entry"]), 2)
            if notify:
                notify(t)
            break


# ----------------------------------------------------------------- stats
def tzdate(iso):
    return pd.Timestamp(iso).tz_convert(TZ).date()


def stats(ts):
    n = len(ts)
    tp = sum(t["status"] == "tp" for t in ts)
    return {"n": n, "tp": tp, "sl": n - tp,
            "net": round(sum(t["r"] for t in ts), 2),
            "pts": round(sum(t["pts"] for t in ts), 2),
            "wr": round(100 * tp / n) if n else 0}


def fmt_r(v):
    return f"{v:+.1f}R"


# ---------------------------------------------------------------- images
BG, CARD, FG, MUTED = "#0f1420", "#1a2133", "#ece9e0", "#8b93a7"
GREEN, RED, GOLD = "#26a69a", "#ef5350", "#d4a017"


def col(v):
    return GREEN if v > 0 else RED if v < 0 else MUTED


def _canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _header(ax, w, h, title, subtitle, s):
    ax.text(0.4, h - 0.55, f"{NAME} · {title}", color=GOLD, fontsize=15, fontweight="bold", va="center")
    ax.text(0.4, h - 0.92, subtitle, color=MUTED, fontsize=10, va="center")
    ax.text(0.4, h - 1.75, fmt_r(s["net"]), color=col(s["net"]), fontsize=34, fontweight="bold", va="center")
    ax.text(0.4, h - 2.3, f"{s['pts']:+.2f} points", color=MUTED, fontsize=10, va="center")
    bw, gap = 1.55, 0.12
    items = [("Trades", s["n"], FG), ("TP", s["tp"], GREEN), ("SL", s["sl"], RED), ("Win rate", f"{s['wr']}%", FG)]
    for i, (lab, val, c) in enumerate(items):
        x = 0.32 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, h - 3.45), bw, 0.85, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=CARD, ec="none"))
        ax.text(x + bw / 2, h - 3.0, str(val), color=c, fontsize=16, fontweight="bold", ha="center", va="center")
        ax.text(x + bw / 2, h - 3.3, lab, color=MUTED, fontsize=8, ha="center", va="center")


def daily_image(day, ts, open_n, path):
    s = stats(ts)
    rows = sorted(ts, key=lambda t: t["close"])
    shown = rows[:16]
    w = 7.2
    h = 5.3 + 0.36 * len(shown) + (0.3 if len(rows) > 16 else 0)
    fig, ax = _canvas(w, h)
    _header(ax, w, h, "Daily Report", f"{day:%A, %d %B %Y}", s)
    y = h - 3.95
    for x, lab, ha in ((0.45, "Time", "left"), (1.45, "Side", "left"), (2.4, "Entry", "left"),
                       (4.0, "Result", "left"), (5.5, "R", "right"), (6.8, "Points", "right")):
        ax.text(x, y, lab, color=MUTED, fontsize=8, ha=ha, va="center")
    for t in shown:
        y -= 0.36
        ax.plot([0.4, w - 0.4], [y + 0.18, y + 0.18], color=CARD, lw=0.8)
        c = col(t["r"])
        close = pd.Timestamp(t["close"]).tz_convert(TZ)
        ax.text(0.45, y, f"{close:%H:%M}", color=FG, fontsize=9, va="center")
        ax.text(1.45, y, SIDE[t["side"]], color=FG, fontsize=9, va="center")
        ax.text(2.4, y, f"{t['entry']}", color=FG, fontsize=9, va="center")
        ax.text(4.0, y, "TP" if t["status"] == "tp" else "SL", color=c, fontsize=9, fontweight="bold", va="center")
        ax.text(5.5, y, fmt_r(t["r"]), color=c, fontsize=9, ha="right", va="center")
        ax.text(6.8, y, f"{t['pts']:+.2f}", color=c, fontsize=9, ha="right", va="center")
    if len(rows) > 16:
        y -= 0.36
        ax.text(0.45, y, f"+ {len(rows) - 16} more", color=MUTED, fontsize=8, va="center")
    ax.text(0.4, 0.3, f"Open trades: {open_n}   ·   Times in Iran time (UTC+3:30)", color=MUTED, fontsize=8, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def bars_image(title, subtitle, labels, subs, vals, s, path, footer):
    """vals: net R per bucket, or None when the bucket had no trades."""
    w, h = 7.2, 8.0
    fig, ax = _canvas(w, h)
    _header(ax, w, h, title, subtitle, s)
    ax2 = fig.add_axes([0.08, 0.13, 0.86, 0.36], facecolor=BG)
    real = [v for v in vals if v is not None]
    top = max([v for v in real] + [1]) * 1.4
    bot = min([v for v in real] + [0]) * 1.4
    for i, v in enumerate(vals):
        if v is None:
            ax2.text(i, 0, "–", color=MUTED, ha="center", va="center", fontsize=12)
            continue
        ax2.bar(i, v, width=0.62, color=col(v), zorder=3)
        ax2.text(i, v + (0.03 * top if v >= 0 else 0.03 * bot), fmt_r(v), color=col(v), fontsize=9,
                 fontweight="bold", ha="center", va="bottom" if v >= 0 else "top")
    ax2.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax2.set_ylim(bot if bot < 0 else -0.15 * top, top)
    ax2.set_xlim(-0.6, len(vals) - 0.4)
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels([f"{a}\n{b}" for a, b in zip(labels, subs)], color=FG, fontsize=8)
    ax2.tick_params(axis="x", length=0)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax2.set_ylabel("Net R", color=MUTED, fontsize=8)
    ax2.grid(axis="y", color=CARD, lw=0.8, zorder=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax.text(0.4, 0.3, footer, color=MUTED, fontsize=8, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def weekly_image(start, end, ts, path):
    names = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
    days = [start + datetime.timedelta(days=i) for i in range(7)]
    vals, subs = [], []
    for d in days:
        dt_ = [t for t in ts if tzdate(t["close"]) == d]
        vals.append(round(sum(t["r"] for t in dt_), 2) if dt_ else None)
        subs.append(f"{d:%d %b}")
    return bars_image("Weekly Report", f"{start:%d %b} – {end:%d %b %Y}", names, subs, vals, stats(ts), path,
                      "Net result per day (R = risk units)   ·   Iran time")


def month_weeks(year, month, until):
    """Saturday-to-Friday weeks of the month up to `until` -> list of (first_day, last_day)."""
    d = datetime.date(year, month, 1)
    out = []
    while d.month == month and d <= until:
        e = d
        while e.weekday() != 4 and (e + datetime.timedelta(days=1)).month == month and e < until:
            e += datetime.timedelta(days=1)
        out.append((d, e))
        d = e + datetime.timedelta(days=1)
    return out


def monthly_image(year, month, until, ts, path):
    weeks = month_weeks(year, month, until)
    vals, labels, subs = [], [], []
    for i, (a, b) in enumerate(weeks, 1):
        wt = [t for t in ts if a <= tzdate(t["close"]) <= b]
        vals.append(round(sum(t["r"] for t in wt), 2) if wt else None)
        labels.append(f"Week {i}")
        subs.append(f"{a.day}-{b.day}")
    return bars_image("Monthly Report", f"{datetime.date(year, month, 1):%B %Y}", labels, subs, vals,
                      stats(ts), path, "Net result per week (R = risk units)   ·   Iran time")


# --------------------------------------------------------------- reports
def send_reports(st, now_utc, send_photo=tg_photo, ready=None):
    ready = tg_ready() if ready is None else ready
    if not ready:
        return
    now = now_utc.astimezone(TZ)
    if now.hour < REPORT_HOUR:
        return
    d = now.date()
    sent = st.setdefault("sent", {})
    done = [t for t in st["trades"] if t["status"] in ("tp", "sl")]
    open_n = sum(t["status"] == "open" for t in st["trades"])

    # daily
    if sent.get("daily") != d.isoformat():
        ts = [t for t in done if tzdate(t["close"]) == d]
        if ts:
            s = stats(ts)
            p = daily_image(d, ts, open_n, "report_daily.png")
            cap = CAP_DAILY.format(date=f"{d:%a %d %b}", tp=s["tp"], sl=s["sl"], net=fmt_r(s["net"]))
            if send_photo(p, cap) is not None:
                sent["daily"] = d.isoformat()

    if d.weekday() != 4:      # Friday only
        return

    # weekly (Saturday .. Friday)
    if sent.get("weekly") != d.isoformat():
        start = d - datetime.timedelta(days=6)
        ts = [t for t in done if start <= tzdate(t["close"]) <= d]
        if ts:
            s = stats(ts)
            p = weekly_image(start, d, ts, "report_weekly.png")
            cap = CAP_WEEKLY.format(date=f"{start:%d %b} – {d:%d %b}", tp=s["tp"], sl=s["sl"], net=fmt_r(s["net"]))
            if send_photo(p, cap) is not None:
                sent["weekly"] = d.isoformat()

    # monthly: only on the last Friday of the month
    last_friday = (d + datetime.timedelta(days=7)).month != d.month
    key = d.strftime("%Y-%m")
    if last_friday and sent.get("monthly") != key:
        ts = [t for t in done if tzdate(t["close"]).year == d.year and tzdate(t["close"]).month == d.month
              and tzdate(t["close"]) <= d]
        if ts:
            s = stats(ts)
            p = monthly_image(d.year, d.month, d, ts, "report_monthly.png")
            cap = CAP_MONTHLY.format(date=f"{d:%B %Y}", tp=s["tp"], sl=s["sl"], net=fmt_r(s["net"]))
            if send_photo(p, cap) is not None:
                sent["monthly"] = key


# ---------------------------------------------------------------- signal
def make_chart(df, o, ema20, ema50, path="chart.png"):
    d = df.tail(80)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for x, (_, r) in enumerate(d.iterrows()):
        c = "#26a69a" if r["Close"] >= r["Open"] else "#ef5350"
        ax.vlines(x, r["Low"], r["High"], color=c, linewidth=1)
        ax.vlines(x, r["Open"], r["Close"], color=c, linewidth=4)
    ax.plot(range(len(d)), ema20.tail(80).values, color="#3b82f6", linewidth=1, label="EMA20")
    ax.plot(range(len(d)), ema50.tail(80).values, color="#a855f7", linewidth=1, label="EMA50")
    for key, color in (("entry", "#d4a017"), ("tp", "#26a69a"), ("sl", "#ef5350")):
        ax.axhline(o[key], color=color, linestyle="--", linewidth=1)
        ax.text(len(d) - 1, o[key], f" {key.upper()} {o[key]}", color=color, va="bottom", ha="right", fontsize=8)
    ax.set_title(f"{NAME} {SIDE[o['signal']]}  ({INTERVAL})", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def compute_signal(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    v1 = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
    r = float(rsi.iloc[-1])
    v2 = 1 if r < 30 else -1 if r > 70 else 0
    v3 = 1 if hist.iloc[-1] > 0 else -1
    score = v1 + v2 + v3
    price, a = float(c.iloc[-1]), float(atr.iloc[-1])

    out = {
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "price": round(price, 2), "rsi": round(r, 1), "atr": round(a, 2),
        "ema20": round(float(ema20.iloc[-1]), 2), "ema50": round(float(ema50.iloc[-1]), 2),
        "hist": round(float(hist.iloc[-1]), 3),
        "votes": [v1, v2, v3], "score": score, "signal": "HOLD",
        "closes": [round(float(x), 2) for x in c.tail(80)],
        "e20": [round(float(x), 2) for x in ema20.tail(80)],
        "e50": [round(float(x), 2) for x in ema50.tail(80)],
    }
    if score >= 2:
        out.update(signal="BUY", entry=round(price, 2), sl=round(price - ATR_SL * a, 2), tp=round(price + ATR_TP * a, 2))
    elif score <= -2:
        out.update(signal="SELL", entry=round(price, 2), sl=round(price + ATR_SL * a, 2), tp=round(price - ATR_TP * a, 2))
    return out, ema20, ema50


def notify_result(t):
    text = MSG_TP.format(r=f"+{t['r']:g}") if t["status"] == "tp" else MSG_SL
    if not t.get("msg_id"):
        text = f"{NAME} {SIDE[t['side']]} {t['entry']}: {text}"
    tg_text(text, reply_to=t.get("msg_id"))


def main():
    import yfinance as yf
    df = yf.download(SYMBOL, period="5d", interval=INTERVAL, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    out, ema20, ema50 = compute_signal(df)

    try:
        prev = json.load(open("signal.json")).get("signal")   # avoid repeating the same alert
    except Exception:
        prev = None
    with open("signal.json", "w") as f:
        json.dump(out, f)

    st = load_state()
    update_trades(st, df, notify_result)           # 1) TP / SL results of open trades
    save_state(st)

    if out["signal"] != "HOLD" and out["signal"] != prev:   # 2) new signal
        idx = df.index
        ts = (idx[-1].tz_localize("UTC") if idx.tz is None else idx[-1].tz_convert("UTC"))
        caption = TEMPLATE.format(name=NAME, side=SIDE[out["signal"]],
                                  entry=out["entry"], tp=out["tp"], sl=out["sl"])
        mid = tg_photo(make_chart(df, out, ema20, ema50), caption) if tg_ready() else None
        st["trades"].append(new_trade(out, ts, mid))
        save_state(st)

    send_reports(st, datetime.datetime.now(datetime.timezone.utc))   # 3) daily / weekly / monthly
    save_state(st)


if __name__ == "__main__":
    main()
