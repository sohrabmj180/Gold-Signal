"""Runs once (by GitHub Actions), computes the signal and writes signal.json."""
import json, datetime, os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

SYMBOL, INTERVAL = "GC=F", "5m"
ATR_SL, ATR_TP = 1.5, 3.0

# ---- Message format: edit this freely ----
NAME = "XAUUSD"
TEMPLATE = "{name} {side}\nEntry: {entry}\nTP: {tp}\nSL: {sl}"
SIDE = {"BUY": "Buy", "SELL": "Sell"}

df = yf.download(SYMBOL, period="5d", interval=INTERVAL, progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna()
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

# previous signal (to avoid repeating the same alert)
try:
    prev = json.load(open("signal.json")).get("signal")
except Exception:
    prev = None

with open("signal.json", "w") as f:
    json.dump(out, f)

def make_chart(df, o, path="chart.png"):
    d = df.tail(80)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for x, (_, r) in enumerate(d.iterrows()):
        col = "#26a69a" if r["Close"] >= r["Open"] else "#ef5350"
        ax.vlines(x, r["Low"], r["High"], color=col, linewidth=1)
        ax.vlines(x, r["Open"], r["Close"], color=col, linewidth=4)
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

def telegram_photo(path, caption):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if not (tok and chat):
        print("Telegram not configured")
        return
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendPhoto",
                          data={"chat_id": chat, "caption": caption},
                          files={"photo": f}, timeout=30)
    print("Telegram:", r.status_code, r.text[:100])

if out["signal"] != "HOLD" and out["signal"] != prev:
    caption = TEMPLATE.format(name=NAME, side=SIDE[out["signal"]],
                              entry=out["entry"], tp=out["tp"], sl=out["sl"])
    telegram_photo(make_chart(df, out), caption)
