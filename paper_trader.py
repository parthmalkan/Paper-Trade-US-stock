"""
paper_trader.py â Paper-trading engine for the US stock model.

WHAT IT DOES
------------
Simulates real trading decisions against a virtual account, so you can see how
the rules behave before risking money. It reads the same scoring logic as
dashboard.py, applies its own cash and position bookkeeping, and writes a
trade log you can inspect.

RULES IMPLEMENTED (all editable in SETTINGS below)
--------------------------------------------------
1. A virtual account starts with a cash balance.
2. Every time you run it, it checks the market timing score.
3. It only invests when the timing score is at or above MIN_TIMING_SCORE
   (default 80 â "80%+ favourable", which is what you asked for).
4. When it does invest, it deploys the contribution across the picks from
   the chosen approach, with a per-name cap.
5. It never invests more cash than the account holds.
6. Every action is written to a CSV trade log with a timestamp.

HONEST LIMITS â READ THESE
--------------------------
- This is a SIMULATION. It places no real orders and touches no broker.
- It runs when you run it. There is no background process, no schedule.
- Prices come from Yahoo's free feed, which can be delayed or wrong.
- The timing score is a RULE, not a probability. An 80 score does not mean
  an 80% chance of gains; it means the VIX and trend inputs scored 80.
- A high timing threshold means long stretches with no investment at all.
  That is the honest consequence of the rule you asked for.
- Not financial advice.

USAGE
-----
    python paper_trader.py                 # run once, apply the rules
    python paper_trader.py --report        # show the account, no new trade
    python paper_trader.py --reset         # wipe the account and start again

State lives in paper_account.json next to this file.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS â change these
# ----------------------------------------------------------------------------
# Each approach gets its own separate account, own cash, own rules.
STARTING_CASH = 100000.0       # starting balance PER APPROACH

# Timing gate
MIN_TIMING_SCORE = 80          # "market must be 80%+ favourable" to invest at all

# --- How often it may trade (per approach) ---
MIN_HOURS_BETWEEN_TRADES = 24  # at most one trade per 24 hours
BETTER_SCORE_WINDOW_HOURS = 72 # inside 72h, only trade if the score is HIGHER

# --- How much gets invested, per approach ---
MONTHLY_PURCHASE_LIMIT = 15000.0   # hard ceiling on buys per calendar month
RUNS_PER_MONTH_EXPECTED = 20       # sizes each run's share of the budget
SCORE_BAND_FULL = 90           # at/above this score, use the full per-run share
SCORE_BAND_MIN = 0.60          # the smallest fraction ever deployed

# --- Portfolio construction (same for every approach) ---
MAX_WEIGHT = 0.20              # cap per stock
TOP_N = 8                      # names per trade
RISK_FREE = 0.045              # for the Sharpe calculation
OPT_MAX_WEIGHT = 0.15
OPT_CANDIDATES = 20

# The three approaches, each with its own account files.
APPROACHES = [
    ("Approach 1", "paper_account_a1.json", "paper_trades_a1.csv"),
    ("Approach 2", "paper_account_a2.json", "paper_trades_a2.csv"),
    ("Approach 3", "paper_account_a3.json", "paper_trades_a3.csv"),
]

# Kept for backwards compatibility with older state files
STATE_FILE = "paper_account.json"
LOG_FILE = "paper_trades.csv"

# Universe: the US bundled list. Kept short here on purpose so a paper run is
# fast; copy more lines in from dashboard.py if you want the full 416.
UNIVERSE = [
    ("AAPL", "Apple Inc.", "Technology", "core"),
    ("MSFT", "Microsoft Corp.", "Technology", "core"),
    ("GOOGL", "Alphabet Inc.", "Technology", "core"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "core"),
    ("NVDA", "NVIDIA Corp.", "Technology", "core"),
    ("META", "Meta Platforms Inc.", "Communication Services", "core"),
    ("AVGO", "Broadcom Inc.", "Technology", "core"),
    ("JPM", "JPMorgan Chase & Co.", "Financials", "core"),
    ("V", "Visa Inc.", "Financials", "core"),
    ("MA", "Mastercard Inc.", "Financials", "core"),
    ("UNH", "UnitedHealth Group", "Health Care", "core"),
    ("LLY", "Eli Lilly & Co.", "Health Care", "core"),
    ("JNJ", "Johnson & Johnson", "Health Care", "core"),
    ("XOM", "Exxon Mobil Corp.", "Energy", "core"),
    ("PG", "Procter & Gamble", "Consumer Staples", "core"),
    ("WMT", "Walmart Inc.", "Consumer Staples", "core"),
    ("COST", "Costco Wholesale", "Consumer Staples", "core"),
    ("HD", "Home Depot Inc.", "Consumer Discretionary", "core"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary", "core"),
    ("PLTR", "Palantir Technologies", "Technology", "momentum"),
    ("MU", "Micron Technology", "Technology", "momentum"),
    ("SMCI", "Super Micro Computer", "Technology", "momentum"),
    ("STX", "Seagate Technology", "Technology", "momentum"),
]


# ----------------------------------------------------------------------------
# 1. MARKET TIMING â same rule as the dashboard, scored 0-100
# ----------------------------------------------------------------------------
def timing_score():
    try:
        vix = yf.Ticker("^INDIAVIX" if False else "^VIX").history(period="5d")["Close"].iloc[-1]
        spx = yf.Ticker("^GSPC").history(period="260d")["Close"]
        now = spx.iloc[-1]
        ma50 = spx.rolling(50).mean().iloc[-1]
        ma200 = spx.rolling(200).mean().iloc[-1]
    except Exception as e:
        return None, f"Could not read market indicators: {e}"

    score = 50
    score += 20 if (now > ma50 and now > ma200) else -20
    score += 15 if vix < 20 else (0 if vix < 30 else -15)
    score = max(0, min(100, score))

    detail = {
        "vix": round(float(vix), 2),
        "sp500": round(float(now), 2),
        "above_50d": bool(now > ma50),
        "above_200d": bool(now > ma200),
        "score": int(score),
    }
    return detail, None


# ----------------------------------------------------------------------------
# 2. SCORING â same logic as the dashboard
# ----------------------------------------------------------------------------
def score_universe():
    tickers = [u[0] for u in UNIVERSE]
    meta = {u[0]: {"company": u[1], "sector": u[2], "bucket": u[3]} for u in UNIVERSE}

    raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)

    rows, closes = [], {}
    for t in tickers:
        try:
            px = raw[t]["Close"].dropna()
            if len(px) < 200 or float(px.iloc[-1]) < 5:
                continue
            closes[t] = px
            rows.append({
                "Ticker": t,
                "Company": meta[t]["company"],
                "Sector": meta[t]["sector"],
                "Bucket": meta[t]["bucket"],
                "Price": float(px.iloc[-1]),
                "Return 12mo": float(px.iloc[-1] / px.iloc[0] - 1),
                "Return 6mo": float(px.iloc[-1] / px.iloc[-126] - 1),
                "Volatility": float(px.pct_change().std() * np.sqrt(252)),
                "Trend": int(px.iloc[-1] > px.rolling(50).mean().iloc[-1])
                       + int(px.iloc[-1] > px.rolling(200).mean().iloc[-1]),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    df["Momentum Score"] = (df["Return 12mo"].rank(pct=True) * 0.6 +
                            df["Return 6mo"].rank(pct=True) * 0.4) * 100
    df["Trend Score"] = df["Trend"] * 50
    df["Low-Vol Score"] = (1 - df["Volatility"].rank(pct=True)) * 100
    df["Composite Score"] = (df["Momentum Score"] * 0.35 +
                             df["Trend Score"] * 0.25 +
                             df["Low-Vol Score"] * 0.20 + 50 * 0.20)

    return df, pd.DataFrame(closes)


def build_target_weights(df, prices, APPROACH=None):
    """Return {ticker: weight} for the chosen approach."""
    APPROACH = APPROACH or "Approach 1"
    if APPROACH == "Approach 1":
        sub = df[df["Bucket"] == "core"].copy()
        if len(sub) < TOP_N:
            sub = df.copy()
        sub = sub.nlargest(TOP_N, "Composite Score")
        w = sub["Composite Score"] / sub["Composite Score"].sum()
        sub["Weight"] = np.minimum(w, MAX_WEIGHT)
        sub["Weight"] = sub["Weight"] / sub["Weight"].sum()
        return dict(zip(sub["Ticker"], sub["Weight"]))

    if APPROACH == "Approach 2":
        sub = df[df["Bucket"] == "momentum"].copy()
        if len(sub) < TOP_N:
            sub = df.copy()
        sub = sub.nlargest(TOP_N, "Return 12mo")
        return {t: 1.0 / len(sub) for t in sub["Ticker"]}

    if APPROACH == "Approach 3":
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            shortlist = df.nlargest(OPT_CANDIDATES, "Composite Score")["Ticker"].tolist()
            sub = prices[[t for t in shortlist if t in prices.columns]].dropna()
            if sub.shape[1] < 2 or len(sub) < 60:
                return {}
            mu = expected_returns.mean_historical_return(sub)
            S = risk_models.sample_cov(sub)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
            ef.max_sharpe(risk_free_rate=RISK_FREE)
            clean = {k: v for k, v in ef.clean_weights().items() if v and v > 0.001}
            total = sum(clean.values()) or 1.0
            return {k: v / total for k, v in clean.items()}
        except Exception:
            return {}

    # Blended 80/20
    core = build_core(df)
    mom = build_mom(df)
    out = {}
    for t, w in core.items():
        out[t] = out.get(t, 0) + w * 0.8
    for t, w in mom.items():
        out[t] = out.get(t, 0) + w * 0.2
    s = sum(out.values()) or 1.0
    return {t: w / s for t, w in out.items()}


def build_core(df):
    sub = df[df["Bucket"] == "core"].copy()
    if len(sub) < TOP_N:
        sub = df.copy()
    sub = sub.nlargest(TOP_N, "Composite Score")
    w = sub["Composite Score"] / sub["Composite Score"].sum()
    return dict(zip(sub["Ticker"], np.minimum(w, MAX_WEIGHT)))


def build_mom(df):
    sub = df[df["Bucket"] == "momentum"].copy()
    if len(sub) < TOP_N:
        sub = df.copy()
    sub = sub.nlargest(TOP_N, "Return 12mo")
    return {t: 1.0 / len(sub) for t in sub["Ticker"]}


# ----------------------------------------------------------------------------
# 3. ACCOUNT STATE
# ----------------------------------------------------------------------------
def load_state(path):
    """Load one account. Starts a fresh one if the file does not exist."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                st = json.load(f)
            st.setdefault("positions", {})
            st.setdefault("runs", 0)
            st.setdefault("invested", 0.0)
            st.setdefault("month", "")
            st.setdefault("month_spent", 0.0)
            st.setdefault("last_trade_utc", "")
            st.setdefault("last_trade_score", 0)
            return st
        except Exception:
            pass
    return {"cash": STARTING_CASH, "positions": {}, "runs": 0, "invested": 0.0,
            "month": "", "month_spent": 0.0, "last_trade_utc": "",
            "last_trade_score": 0}


def save_state(state, path):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def log_trades(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def hours_since(iso_utc):
    """Hours elapsed since a stored UTC timestamp, or None if unset."""
    if not iso_utc:
        return None
    try:
        t = datetime.strptime(iso_utc, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
    except Exception:
        return None


def trade_allowed(state, score):
    """The two frequency rules, applied per approach.

    Rule 1: never more than one trade per MIN_HOURS_BETWEEN_TRADES (24h).
    Rule 2: a second trade inside BETTER_SCORE_WINDOW_HOURS (72h) is only
            allowed if the timing score beats the score of the last trade."""
    elapsed = hours_since(state.get("last_trade_utc", ""))
    if elapsed is None:
        return True, "no previous trade"

    if elapsed < MIN_HOURS_BETWEEN_TRADES:
        wait = MIN_HOURS_BETWEEN_TRADES - elapsed
        return False, f"only {elapsed:.1f}h since the last trade (wait {wait:.1f}h more)"

    if elapsed < BETTER_SCORE_WINDOW_HOURS:
        prev = float(state.get("last_trade_score", 0))
        if score <= prev:
            return False, (f"inside the {BETTER_SCORE_WINDOW_HOURS}h window and the "
                           f"score {score} does not beat the last trade score {prev:.0f}")
        return True, f"score {score} beats the last trade score {prev:.0f}"

    return True, f"{elapsed:.0f}h since the last trade",


def account_value(state, price_map):
    total = float(state.get("cash", 0.0))
    for t, p in state.get("positions", {}).items():
        total += p.get("shares", 0.0) * float(price_map.get(t, p.get("avg_price", 0.0)))
    return total

def run_approach(approach, state_file, log_file, timing, now_utc):
    """Apply the rules for ONE approach, on its own account."""
    state = load_state(state_file)
    score = timing["score"]

    state["runs"] = state.get("runs", 0) + 1

    # --- Rule gate 1: timing must be favourable ---
    if score < MIN_TIMING_SCORE:
        save_state(state, state_file)
        return f"{approach}: score {score} is below the {MIN_TIMING_SCORE} gate. No trade."

    # --- Rule gate 2: at most one trade per 24h ---
    # --- Rule gate 3: inside 72h, only if the score beats the last trade ---
    allowed, why = trade_allowed(state, score)
    if not allowed:
        save_state(state, state_file)
        return f"{approach}: {why}. No trade."

    # --- monthly budget ---
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    if state.get("month") != month_key:
        state["month"] = month_key
        state["month_spent"] = 0.0

    spent = float(state.get("month_spent", 0.0))
    remaining = MONTHLY_PURCHASE_LIMIT - spent
    if remaining <= 0:
        save_state(state, state_file)
        return (f"{approach}: monthly limit used (${spent:,.2f} of "
                f"${MONTHLY_PURCHASE_LIMIT:,.2f}). No trade.")

    if state["cash"] <= 0:
        save_state(state, state_file)
        return f"{approach}: no cash left. No trade."

    # --- size from the score ---
    per_run = MONTHLY_PURCHASE_LIMIT / max(1, RUNS_PER_MONTH_EXPECTED)
    if score >= SCORE_BAND_FULL:
        band = 1.0
    else:
        span = max(1, SCORE_BAND_FULL - MIN_TIMING_SCORE)
        band = SCORE_BAND_MIN + (1.0 - SCORE_BAND_MIN) * ((score - MIN_TIMING_SCORE) / span)
        band = max(SCORE_BAND_MIN, min(1.0, band))
    to_invest = min(per_run * band, remaining, state["cash"])

    df, prices = score_universe()
    if df.empty:
        save_state(state, state_file)
        return f"{approach}: could not score the universe. No trade."

    weights = build_target_weights(df, prices, approach)
    if not weights:
        save_state(state, state_file)
        return f"{approach}: could not build weights. No trade."

    price_map = dict(zip(df["Ticker"], df["Price"]))
    rows, deployed = [], 0.0
    cash_left = state["cash"]

    for tk, w in sorted(weights.items(), key=lambda x: -x[1]):
        amount = min(to_invest * w, cash_left)
        if amount < 1 or tk not in price_map:
            continue
        price = price_map[tk]
        shares = amount / price
        cash_left -= amount
        deployed += amount

        pos = state["positions"].get(tk, {"shares": 0.0, "avg_price": 0.0})
        new_shares = pos["shares"] + shares
        new_cost = pos["shares"] * pos["avg_price"] + amount
        state["positions"][tk] = {
            "shares": round(new_shares, 6),
            "avg_price": round(new_cost / new_shares, 4) if new_shares else 0.0,
        }

        rows.append({
            "timestamp_utc": now_utc,
            "action": "BUY",
            "ticker": tk,
            "shares": round(shares, 6),
            "price": round(price, 4),
            "amount": round(amount, 2),
            "timing_score": score,
            "approach": approach,
        })

    if deployed <= 0:
        save_state(state, state_file)
        return f"{approach}: nothing investable. No trade."

    state["cash"] = round(state["cash"] - deployed, 4)
    state["invested"] = round(state.get("invested", 0.0) + deployed, 2)
    state["month_spent"] = round(spent + deployed, 2)
    state["last_trade_utc"] = now_utc
    state["last_trade_score"] = score
    save_state(state, state_file)
    log_trades(rows, log_file)

    return (f"{approach}: BOUGHT ${deployed:,.2f} across {len(rows)} names "
            f"at score {score} ({why}).")


def run_once():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nPaper run at {now_utc}")

    timing, err = timing_score()
    if timing is None:
        print(f"  {err}")
        return

    print(f"  Timing score: {timing['score']}/100  (VIX {timing['vix']})")

    results = []
    for approach, state_file, log_file in APPROACHES:
        msg = run_approach(approach, state_file, log_file, timing, now_utc)
        print("  " + msg)
        results.append(msg)

    traded = [r for r in results if "BOUGHT" in r]
    if traded:
        print(f"\n  {len(traded)} approach(es) traded this run.")
    else:
        print("\n  No approach traded this run.")


def write_summary():
    """Rank the three approaches by total return, with a plain-language reason."""
    try:
        df, _ = score_universe()
    except Exception:
        df = None
    price_map = dict(zip(df["Ticker"], df["Price"])) if df is not None and not df.empty else {}

    rows = []
    for approach, state_file, log_file in APPROACHES:
        st = load_state(state_file)
        val = account_value(st, price_map)
        ret = (val / STARTING_CASH - 1) * 100 if STARTING_CASH else 0.0
        trades = 0
        if os.path.exists(log_file):
            try:
                trades = len(pd.read_csv(log_file))
            except Exception:
                trades = 0
        rows.append({
            "Approach": approach,
            "Total value": round(val, 2),
            "Return (%)": round(ret, 2),
            "Cash": round(st.get("cash", 0.0), 2),
            "Positions": len(st.get("positions", {})),
            "Trades": trades,
            "Last trade": st.get("last_trade_utc", "") or "-",
        })

    summary = pd.DataFrame(rows).sort_values("Return (%)", ascending=False)
    summary.to_csv("paper_summary.csv", index=False)

    best = summary.iloc[0] if len(summary) else None
    if best is not None:
        print("\n" + "=" * 62)
        print("SUMMARY - best approach to date")
        print("=" * 62)
        for _, r in summary.iterrows():
            print(f"  {r['Approach']:<12} ${r['Total value']:>12,.2f}  "
                  f"{r['Return (%)']:>+7.2f}%  ({r['Trades']} trades)")
        print(f"\n  Leader: {best['Approach']} at {best['Return (%)']:+.2f}%.")
        print("  Reason: it leads on total return, which blends the picks it")
        print("  bought with the cash it left idle. See paper_summary.csv.")
        print("=" * 62)


def main():
    p = argparse.ArgumentParser(description="Paper-trader for the US stock model")
    p.add_argument("--report", action="store_true", help="show the account, make no trade")
    p.add_argument("--reset", action="store_true", help="wipe the account and start over")
    args = p.parse_args()

    if args.reset:
        for _, sf, lf in APPROACHES:
            for p in (sf, lf):
                if os.path.exists(p):
                    os.remove(p)
        if os.path.exists("paper_summary.csv"):
            os.remove("paper_summary.csv")
        print("Account reset. Starting fresh.")
        return

    if args.report:
        write_summary()
        return

    run_once()
    write_summary()


if __name__ == "__main__":
    main()
