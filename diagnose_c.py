"""
diagnose_c.py - READ ONLY. Why does setup C (Smart Money) almost never signal?  Counts only, no win rate / profit.
Run from GitHub Actions (workflow "DiagC"): needs the secret TD_KEY. Writes diagnosis_c.txt.
"""
import os, sys
import numpy as np
import pandas as pd
import config as C
import strategy as S
import fib as F
import smc as M
import joint as J

STAGES = ["few_pivots", "not_bearish_before", "no_higher_low", "no_choch", "no_bos", "ok"]
STAGE_TEXT = {"few_pivots": "کمتر از ۶ سقف/کف متناوب", "not_bearish_before": "ساختار قبلی نزولی نبود (سقف و کف پایین‌تر)",
              "no_higher_low": "کف بالاتر (HL) تشکیل نشد", "no_choch": "CHoCH نشد (close بالای آخرین سقف پایین‌تر)",
              "no_bos": "BOS نشد (close بالای سقف جدید)", "ok": "ساختار کامل بود"}
REASONS = ["no_data", "no_session", "volatility_spike", "no_bias", "no_structure", "no_ob", "stale_ob", "ob_broken",
           "no_retest", "ob_touched", "no_reaction", "sl_too_wide", "SIGNAL"]

def structure_stage(m, L):
    """Deepest stage any candidate pattern reached (BUY orientation). Mirrors smc.find_structure step by step."""
    alt = F.alternate(F.swing_pivots(m, L))
    cl = m["Close"].values
    best = "few_pivots"
    rank = {s: k for k, s in enumerate(STAGES)}
    for j in range(len(alt) - 1, 4, -1):
        if alt[j][1] != "L":
            continue
        Ha, La, Hb, Lb, Hc, Lc = alt[j - 5: j + 1]
        if not (Ha[1] == "H" and Hb[2] < Ha[2] and Lb[2] < La[2]):
            st = "not_bearish_before"
        elif not (Lc[2] > Lb[2]):
            st = "no_higher_low"
        elif not (Hc[2] > Hb[2] and (cl[Lb[0] + 1: Hc[0] + 1] > Hb[2]).any()):
            st = "no_choch"
        elif len(np.flatnonzero(cl[Lc[0] + 1:] > Hc[2])) == 0:
            st = "no_bos"
        else:
            st = "ok"
        if rank[st] > rank[best]:
            best = st
    return best

def analyze(df):
    first, stage, evaluated = {}, {}, 0
    for i in range(C.MIN_CANDLES - 1, len(df)):
        res = M.evaluate_smc_at(df, i, {}, None)
        evaluated += 1
        key = "SIGNAL" if res["status"] == "SIGNAL" else res["reason"]
        first[key] = first.get(key, 0) + 1
        if key == "no_structure":
            w = df.iloc[max(0, i - C.SMC_LOOKBACK + 1): i + 1]
            m = w[["Open", "High", "Low", "Close"]] if res["side"] == "BUY" else M._mirror(w)
            s = structure_stage(m, C.SMC_PIVOT_L)
            stage[s] = stage.get(s, 0) + 1
    return {"evaluated": evaluated, "first": first, "stage": stage}

VARIANTS = [
    ("فعلی (بدون تغییر)", {}),
    ("SMC_PIVOT_L  3 -> 2", {"SMC_PIVOT_L": 2}),
    ("SMC_PIVOT_L  3 -> 5", {"SMC_PIVOT_L": 5}),
    ("SMC_LOOKBACK  300 -> 600", {"SMC_LOOKBACK": 600}),
    ("SMC_OB_MAX_AGE  48 -> 96", {"SMC_OB_MAX_AGE": 96}),
    ("SMC_REACT_WINDOW  2 -> 4", {"SMC_REACT_WINDOW": 4}),
    ("SMC_OB_BODY_MIN / SMC_REACTION_BODY  0.40 -> 0.30", {"SMC_OB_BODY_MIN": 0.30, "SMC_REACTION_BODY": 0.30}),
    ("R_MAX_ATR  2.0 -> 3.0", {"R_MAX_ATR": 3.0}),
    ("ساعت کاری C = ۵ تا ۲۳ تهران", {"SMC_SESSIONS": (("Asia/Tehran", "05:00", "23:00"),)}),
]

def _count(df):
    trades, info = J.simulate_multi(df, [("C", M.evaluate_smc_at)])
    return info["signals"].get("C", 0)

def sensitivity(df):
    out = []
    for label, change in VARIANTS:
        old = {k: getattr(C, k) for k in change}
        try:
            for k, v in change.items():
                setattr(C, k, v)
            out.append((label, _count(df)))
        finally:
            for k, v in old.items():
                setattr(C, k, v)
    old_bias = M.daily_bias                                   # "no daily bias": allow BOTH directions
    try:
        total = 0
        for side in ("BUY", "SELL"):
            M.daily_bias = lambda d, i, s=side: s
            total += _count(df)
    finally:
        M.daily_bias = old_bias
    out.append(("بدون Daily Bias (هر دو جهت)", total))
    return out

def build_report(df, info, sens):
    t0, t1 = df.index[C.MIN_CANDLES - 1], df.index[-1] + S.TF
    weeks = max((t1 - t0).total_seconds() / 86400 / 7, 1e-9)
    first = info["first"]
    in_win = info["evaluated"] - first.get("no_data", 0) - first.get("no_session", 0)
    L = ["=== DIAGNOSIS of setup C (counts only, no results) ===",
         f"data: {len(df)} candles M15, {C.SOURCE_TYPE}; period {t0.isoformat()} -> {t1.isoformat()} ({weeks:.1f} weeks)",
         f"candles inside the C window (London/New York, weekdays): {in_win}  of {info['evaluated']}", "",
         "--- 1) first failing rule (share of in-window candles) ---"]
    for k in REASONS:
        if k in ("no_data", "no_session"):
            continue
        v = first.get(k, 0)
        L.append(f"{k:17s} {v:6d}  {100 * v / in_win if in_win else 0:5.1f}%")
    L.append(f"(no_data={first.get('no_data', 0)}, outside window={first.get('no_session', 0)})")
    ns = first.get("no_structure", 0)
    if ns:
        L += ["", "--- 2) no_structure: the deepest step the pattern CHoCH -> HL -> BOS reached ---"]
        for k in STAGES:
            v = info["stage"].get(k, 0)
            L.append(f"{v:6d}  {100 * v / ns:5.1f}%  {STAGE_TEXT[k]}")
    L += ["", "--- 3) setup C signals over the whole period (state-aware: one per Tehran day, locks) ---"]
    for label, n in sens:
        L.append(f"{label:52s} {n:4d} signals   {n / weeks:5.2f} per week")
    L += ["", "NOTE: counts only. Win rate / profit are NOT shown on purpose: choosing a setting by its results only fits the past."]
    return "\n".join(L)

def main():
    raw = J.fetch_history(int(os.getenv("JOINT_PAGES", "8")))
    df = S.add_indicators(J.clean_history(raw))
    if len(df) <= C.MIN_CANDLES:
        print(f"not enough candles: {len(df)}")
        return 1
    print("analysing ...", flush=True)
    info = analyze(df)
    print("sensitivity ...", flush=True)
    text = build_report(df, info, sensitivity(df))
    print(text)
    with open("diagnosis_c.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
