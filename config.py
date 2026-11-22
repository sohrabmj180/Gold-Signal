"""All constants of the strategy live here. Nothing is tuned automatically."""

# ---- mode (ONLY the owner changes this by hand; the bot never edits it) ----
MODE = "TEST"                 # "TEST" or "LIVE"

# ---- data source ----
SYMBOL = "GC=F"
SOURCE_TYPE = "FUTURES"       # "SPOT" or "FUTURES"  (GC=F is futures)
PERIOD = "60d"                # Yahoo limit for 15m candles
INTERVAL = "15m"
TIMEFRAME_MIN = 15
PRICE_MODEL = "BID"           # assumption: data treated as Bid

# ---- step 1: data checks ----
MIN_CANDLES = 600
MAX_LAG_CANDLES = 3           # last closed candle older than this -> market_closed
GAP_FACTOR = 2                # gap > 2 x timeframe between consecutive candles
GAP_WINDOW = 7                # candles S-6..S must have no gap

# ---- indicators ----
EMA_PERIODS = (20, 50, 200)
ATR_N = 14
ADX_N = 14

# ---- step 2: strategy ----
ADX_MIN = 20
SLOPE_LOOKBACK = 6            # EMA50 now vs 6 candles ago
PULLBACK_MAX = 5              # touch candle within 5 candles before S
BODY_MIN = 0.40               # body >= 40% of candle range
RANGE_MAX_ATR = 2.5           # volatility_spike
SPREAD_USD = 0.30
SL_BUFFER_ATR = 0.1
R_MIN_ATR = 1.0
R_MAX_ATR = 2.0
RR = 2.0

# sessions: (timezone, start "HH:MM", end "HH:MM"), checked on candle-S close time
SESSIONS = (("Europe/London", "08:00", "12:00"),
            ("America/New_York", "09:00", "13:00"))
FRIDAY_CUTOFF_UTC = "20:00"   # Friday after this: no signals; weekend: no signals

# fixed daily news blackouts (America/New_York, weekdays)
DAILY_BLACKOUTS_NY = (("08:25", "09:05"), ("09:55", "10:35"))
NEWS_BEFORE_MIN = 30
NEWS_AFTER_MIN = 60
NEWS_FILE = "news_blackout.json"
NEWS_STALE_DAYS = 14

# ---- step 3: state ----
COOLDOWN_CANDLES = 8          # after SL, same-direction signals banned
EXPIRY_CANDLES = 32
MARKET_CLOSE_UTC = ("Friday", "21:00")
MAX_SCAN_CANDLES = 16
LATE_MAX_ATR = 0.3
TG_RETRIES = 3

# HOLD reason priority (first failed condition wins)
HOLD_PRIORITY = ("no_data", "market_closed", "no_session", "news_blackout", "volatility_spike",
                 "open_signal", "cooldown", "weak_trend", "low_adx", "no_pullback",
                 "no_trigger", "sl_too_wide")

DISCLAIMER = "این سیگنال‌ها آزمایشی/تست‌نشده‌اند، توصیه مالی نیستند و ریسک ضرر دارند."

# ---- step 4: output files / messages ----
SIGNAL_FILE = "signal.json"
LOG_FILE = "signals_log.csv"
STATE_FILE = "state.json"
ERROR_FILE = "error.json"
NAME = "XAUUSD"
SIDE_LABEL = {"BUY": "Buy", "SELL": "Sell"}
REPORT_HOUR = 22                  # Tehran time
REPORT_EMPTY_DAILY = False        # True: also send a daily report when no signal was closed that day
LOG_COLUMNS = ["id", "time_utc", "side", "reference", "entry", "sl", "tp", "r", "status", "result_r",
               "close_time_utc", "delivery"]

# ---- step 5: backtest (single fixed setup, nothing is tuned or compared) ----
BACKTEST_SPREAD_USD = 0.60        # "high spread" model = 2x the live SPREAD_USD  (my assumption, change by hand)
BACKTEST_MIN_TRADES = 300
BACKTEST_SPLIT = 0.60             # first 60% of the TIME span vs last 40%
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 12345
# ---- Twelve Data (spot): later assignments override the lines above ----
SOURCE_TYPE = "SPOT"
PROVIDER = "TWELVEDATA"
TD_SYMBOL = "XAU/USD"
TD_OUTPUTSIZE = 1000
# ---- Owner change: bot is ON 05:00 -> 23:00 Tehran time (23:00 = off) ----
SESSIONS = (("Asia/Tehran", "05:00", "23:00"),)
# ---- Setup B: Fibonacci pullback inside a swing trend (my draft values, not optimized) ----
FIB_PIVOT_L = 5               # swing = fractal: higher/lower than 5 candles on each side (confirmed 5 candles later)
FIB_LOOKBACK = 400            # candles searched for swings
FIB_LEG_MIN_ATR = 2.0         # impulse leg must be >= 2 ATR
FIB_MIN = 0.382               # correction must retrace at least 38.2% of the leg
FIB_MAX = 0.786               # ... and at most 78.6% (deeper = structure failed)
FIB_FRESH = 6                 # reversal candle must come within 6 candles after the correction extreme

# ---- Two setups: A (EMA pullback) and B (Fibonacci pullback) ----
ENABLE_SETUP_B = True
LOG_COLUMNS = ["id", "time_utc", "side", "reference", "entry", "sl", "tp", "r", "status", "result_r",
               "close_time_utc", "delivery", "setup"]
# ---- Setup C: Smart Money (CHoCH + BOS + Order Block retest) - my reading of the owner's rules ----
ENABLE_SETUP_C = True
SMC_PIVOT_L = 3               # swing = fractal with 3 candles on each side (confirmed 3 candles later)
SMC_LOOKBACK = 300            # candles searched for the structure
SMC_OB_BODY_MIN = 0.40        # "strong" order-block candle: body >= 40% of its range (same 40% as setups A and B)
SMC_OB_MAX_AGE = 48           # the order block must be at most 48 candles (12 hours) old
SMC_REACTION_BODY = 0.40      # reaction candle: body >= 40% of its range (same 40% as setups A and B)
SMC_REACT_WINDOW = 2          # first touch of the zone must be on the signal candle or within 2 candles before it
SMC_BIAS_EMA = 10             # daily bias: last completed UTC day closed above/below the EMA(10) of daily closes
SMC_SESSIONS = (("Europe/London", "08:00", "12:00"), ("America/New_York", "09:00", "13:00"))   # "high volatility hours"
TD_OUTPUTSIZE = 5000          # ~ 54 trading days of M15: enough history for the daily bias
