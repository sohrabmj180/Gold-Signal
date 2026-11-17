"""
Gold signal - GitHub Actions version (runs once every 5 min).
Environment variables: TG_TOKEN, TG_CHAT_ID
Educational tool only - not financial advice.
"""
import os, json, time, datetime as dt
from zoneinfo import ZoneInfo
import requests, pandas as pd, yfinance as yf, jdatetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= settings =================
CANDLE_SYMBOL = "GC=F"          # futures candles, shifted to spot level (see get_candles)
INTERVAL = "5m"
ATR_SL, ATR_TP = 1.5, 3.0       # stop = 1.5 ATR, target = 3 ATR  -> 2R
RR = ATR_TP / ATR_SL
CANDLE_REFRESH = 60             # seconds: re-download candles + indicators
PRICE_REFRESH = 10              # seconds: live price check (TP/SL detection)
MAX_PRICE_GAP = 30              # ignore live price if it differs this much from candles
REPORT_HOUR = 22                # reports at 22:00 Tehran time
NAME = "XAUUSD"
TEMPLATE = "{name} {side}\nEntry: {entry}\nTP: {tp}\nSL: {sl}"
SIDE = {"BUY": "Buy", "SELL": "Sell"}
# ============================================

TZ = ZoneInfo("Asia/Tehran")
DATA_DIR = os.getenv("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
STATE_FILE = os.path.join(DATA_DIR, "state.json")

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception as e:
        print("save error:", e)

TRADES = load_json(TRADES_FILE, [])
STATE = load_json(STATE_FILE, {"last_signal": None, "last_daily": "", "last_weekly": "", "last_monthly": ""})
S = {"df": None, "a": None, "latest": {"signal": "HOLD", "price": 0, "score": 0, "votes": [0, 0, 0],
     "rsi": 0, "atr": 0, "ema20": 0, "ema50": 0, "hist": 0, "closes": [], "e20": [], "e50": [],
     "updated": dt.datetime.now(dt.timezone.utc).isoformat()}}

def utcnow():
    return dt.datetime.now(dt.timezone.utc)

# ---------------- Telegram ----------------
def tg(method, data=None, files=None):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not (tok and chat):
        print("Telegram not configured")
        return None
    data = dict(data or {})
    data["chat_id"] = chat
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/{method}", data=data, files=files, timeout=30)
        j = r.json()
        if not j.get("ok"):
            print("Telegram error:", j)
        return j.get("result")
    except Exception as e:
        print("Telegram exception:", e)
        return None

# ---------------- data + indicators ----------------
def get_candles():
    """Futures candles (GC=F) shifted so they sit on the live SPOT price level."""
    df = yf.download(CANDLE_SYMBOL, period="5d", interval=INTERVAL, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if len(df) < 60:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    spot = live_price(None)
    if spot:
        off = spot - float(df["Close"].iloc[-1])
        if abs(off) <= MAX_PRICE_GAP:
            df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] + off
    return df

def live_price(ref):
    """Spot gold live price. Returns None if unavailable or too far from candles."""
    try:
        p = float(requests.get("https://api.gold-api.com/price/XAU", timeout=8).json()["price"])
        if p > 0 and (ref is None or abs(p - ref) <= MAX_PRICE_GAP):
            return p
    except Exception:
        pass
    return None

def analyze(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    v1 = 1 if e20.iloc[-1] > e50.iloc[-1] else -1
    r = float(rsi.iloc[-1])
    v2 = 1 if r < 30 else -1 if r > 70 else 0
    v3 = 1 if hist.iloc[-1] > 0 else -1
    return {"price": float(c.iloc[-1]), "rsi": round(r, 1), "atr": float(atr.iloc[-1]),
            "ema20": round(float(e20.iloc[-1]), 2), "ema50": round(float(e50.iloc[-1]), 2),
            "hist": round(float(hist.iloc[-1]), 3), "votes": [v1, v2, v3], "score": v1 + v2 + v3,
            "closes": [round(float(x), 2) for x in c.tail(80)],
            "e20": [round(float(x), 2) for x in e20.tail(80)],
            "e50": [round(float(x), 2) for x in e50.tail(80)],
            "e20s": e20, "e50s": e50}

def side_of(score):
    return "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"

# ---------------- trades ----------------
def open_trade():
    for t in TRADES:
        if t["status"] == "open":
            return t
    return None

def make_chart(df, a, tr, path="/tmp/chart.png"):
    d = df.tail(80)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for x, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r["Close"] >= r["Open"] else "#ef5350"
        ax.vlines(x, r["Low"], r["High"], color=col, linewidth=1)
        ax.vlines(x, r["Open"], r["Close"], color=col, linewidth=4)
    ax.plot(range(len(d)), a["e20s"].tail(80).values, color="#3b82f6", linewidth=1, label="EMA20")
    ax.plot(range(len(d)), a["e50s"].tail(80).values, color="#a855f7", linewidth=1, label="EMA50")
    for key, color in (("entry", "#d4a017"), ("tp", "#26a69a"), ("sl", "#ef5350")):
        ax.axhline(tr[key], color=color, linestyle="--", linewidth=1)
        ax.text(len(d) - 1, tr[key], f" {key.upper()} {tr[key]}", color=color, va="bottom", ha="right", fontsize=8)
    lo = min(float(d["Low"].min()), tr["sl"], tr["tp"]); hi = max(float(d["High"].max()), tr["sl"], tr["tp"])
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{NAME} {SIDE[tr['side']]}  ({INTERVAL})", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

def open_new(side, df, a):
    live = live_price(a["price"])
    entry = live if live else a["price"]
    atr = a["atr"]
    if side == "BUY":
        sl, tp = entry - ATR_SL * atr, entry + ATR_TP * atr
    else:
        sl, tp = entry + ATR_SL * atr, entry - ATR_TP * atr
    tr = {"id": int(time.time()), "side": side, "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
          "open_time": utcnow().isoformat(), "status": "open", "msg_id": None}
    caption = TEMPLATE.format(name=NAME, side=SIDE[side], entry=tr["entry"], tp=tr["tp"], sl=tr["sl"])
    with open(make_chart(df, a, tr), "rb") as f:
        res = tg("sendPhoto", {"caption": caption}, {"photo": f})
    if res:
        tr["msg_id"] = res.get("message_id")
    TRADES.append(tr)
    save_json(TRADES_FILE, TRADES)
    print("OPENED", tr)

def close_trade(tr, status):
    tr["status"] = status
    tr["close_time"] = utcnow().isoformat()
    if status == "tp":
        tr["r"], tr["pts"] = RR, round(abs(tr["tp"] - tr["entry"]), 2)
        text = f"Tp ✅ +{RR:g}R"
    else:
        tr["r"], tr["pts"] = -1, -round(abs(tr["entry"] - tr["sl"]), 2)
        text = "SL ❌ -1R"
    save_json(TRADES_FILE, TRADES)
    data = {"text": text}
    if tr.get("msg_id"):
        data["reply_to_message_id"] = tr["msg_id"]
    tg("sendMessage", data)
    print("CLOSED", tr)

def check_price(tr, price):
    if tr["side"] == "BUY":
        hit_tp, hit_sl = price >= tr["tp"], price <= tr["sl"]
    else:
        hit_tp, hit_sl = price <= tr["tp"], price >= tr["sl"]
    if hit_sl:
        close_trade(tr, "sl")
    elif hit_tp:
        close_trade(tr, "tp")

def check_candles(tr, df):
    """Catches TP/SL touched between live-price checks. If both hit in one candle, SL wins (conservative)."""
    opened = pd.Timestamp(tr["open_time"])
    for ts, r in df[df.index > opened].iterrows():
        hi, lo = float(r["High"]), float(r["Low"])
        if tr["side"] == "BUY":
            hit_tp, hit_sl = hi >= tr["tp"], lo <= tr["sl"]
        else:
            hit_tp, hit_sl = lo <= tr["tp"], hi >= tr["sl"]
        if hit_sl:
            return close_trade(tr, "sl")
        if hit_tp:
            return close_trade(tr, "tp")

# ---------------- reports ----------------
def jal(d):
    j = jdatetime.date.fromgregorian(date=d)
    return j.year, j.month, j.day

def closed_where(pred):
    out = []
    for t in TRADES:
        if t["status"] != "open":
            d = dt.datetime.fromisoformat(t["close_time"]).astimezone(TZ).date()
            if pred(d):
                out.append(t)
    return out

def report_text(title, trs):
    if not trs:
        return f"📊 {title}\n\nمعامله‌ی بسته‌شده‌ای نبود."
    w = sum(1 for t in trs if t["status"] == "tp")
    l = len(trs) - w
    net = sum(t["r"] for t in trs)
    pts = sum(t["pts"] for t in trs)
    return (f"📊 {title}\n\nتعداد معاملات: {len(trs)}\n✅ سود: {w}\n❌ ضرر: {l}\n"
            f"نرخ برد: {w * 100 / len(trs):.0f}%\nبرآیند: {net:+g}R ({pts:+.2f} دلار)")

def check_reports():
    now = dt.datetime.now(TZ)
    if now.hour < REPORT_HOUR:
        return
    today = now.date()
    iso = today.isoformat()
    if STATE["last_daily"] != iso:
        tg("sendMessage", {"text": report_text("گزارش روزانه", closed_where(lambda d: d == today))})
        STATE["last_daily"] = iso
        save_json(STATE_FILE, STATE)
    if now.weekday() == 4 and STATE["last_weekly"] != iso:      # Friday
        start = today - dt.timedelta(days=6)                    # Saturday..Friday
        tg("sendMessage", {"text": report_text("گزارش هفتگی", closed_where(lambda d: start <= d <= today))})
        STATE["last_weekly"] = iso
        save_json(STATE_FILE, STATE)
    if jal(today + dt.timedelta(days=1))[2] == 1 and STATE["last_monthly"] != iso:   # last day of Jalali month
        ym = jal(today)[:2]
        tg("sendMessage", {"text": report_text("گزارش ماهانه", closed_where(lambda d: jal(d)[:2] == ym))})
        STATE["last_monthly"] = iso
        save_json(STATE_FILE, STATE)

# ---------------- main logic ----------------
def publish(live=None):
    a = S["a"]
    if not a:
        return
    out = {k: a[k] for k in ("rsi", "ema20", "ema50", "hist", "votes", "score", "closes", "e20", "e50")}
    out.update(updated=utcnow().isoformat(), price=round(live if live else a["price"], 2),
               atr=round(a["atr"], 2), signal="HOLD")
    tr = open_trade()
    if tr:
        out.update(signal=tr["side"], entry=tr["entry"], tp=tr["tp"], sl=tr["sl"])
    S["latest"] = out

def refresh():
    df = get_candles()
    if df is None:
        print("no candles")
        return
    a = analyze(df)
    S["df"], S["a"] = df, a
    tr = open_trade()
    if tr:
        check_candles(tr, df)
    side = side_of(a["score"])
    if STATE["last_signal"] is None:                 # first run: just remember, don't trade
        STATE["last_signal"] = side
        save_json(STATE_FILE, STATE)
    elif side != STATE["last_signal"]:
        if side != "HOLD" and not open_trade():
            open_new(side, df, a)
        STATE["last_signal"] = side
        save_json(STATE_FILE, STATE)
    publish()

def tick():
    a = S["a"]
    if not a:
        return
    live = live_price(a["price"])
    tr = open_trade()
    if tr and live:
        check_price(tr, live)
    publish(live)
    check_reports()

def main():
    try:
        refresh()
        tick()
    except Exception as e:
        print("error:", e)
    # always write files so the workflow can commit them
    save_json(os.path.join(DATA_DIR, "signal.json"), S["latest"])
    save_json(TRADES_FILE, TRADES)
    save_json(STATE_FILE, STATE)

if __name__ == "__main__":
    main()
