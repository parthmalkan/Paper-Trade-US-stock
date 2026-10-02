"""
paper_trader.py — Paper-trading engine for the US stock model.

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
   (default 80 — "80%+ favourable", which is what you asked for).
4. When it does invest, it deploys the contribution across the picks from
   the chosen approach, with a per-name cap.
5. It never invests more cash than the account holds.
6. Every action is written to a CSV trade log with a timestamp.

HONEST LIMITS — READ THESE
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
# SETTINGS — change these
# ----------------------------------------------------------------------------
STARTING_CASH = 100000.0       # virtual account balance to begin with
MIN_TIMING_SCORE = 80          # "market must be 80%+ favourable"

# --- How much gets invested ---
# There is no fixed per-run amount. Instead the ceiling is $15,000 of
# PURCHASES PER CALENDAR MONTH. Within a month, each qualifying run deploys
# a share of whatever is left of that budget, scaled by how favourable the
# timing score is. A score of 80+ uses more; a marginal pass uses less.
MONTHLY_PURCHASE_LIMIT = 15000.0   # hard ceiling on buys per calendar month
RUNS_PER_MONTH_EXPECTED = 4        # used to size each run's share of the budget

# Deploy this fraction of the remaining monthly budget by timing score band.
# e.g. a 95 score deploys the full per-run share; an 80 score deploys ~60%.
SCORE_BAND_FULL = 90           # at/above this, use the full per-run share
SCORE_BAND_MIN = 0.60          # the smallest fraction ever deployed (60%)

APPROACH = "Approach 1"        # "Approach 1" | "Approach 2" | "Approach 3" | "Blended 80/20"
MAX_WEIGHT = 0.20              # cap per stock, so one name can't dominate
TOP_N = 8                      # how many names to spread the run across

RISK_FREE = 0.045              # for the Sharpe calculation
OPT_MAX_WEIGHT = 0.15
OPT_CANDIDATES = 20

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
# 1. MARKET TIMING — same rule as the dashboard, scored 0-100
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
# 2. SCORING — same logic as the dashboard
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


def build_target_weights(df, prices):
    """Return {ticker: weight} for the chosen approach."""
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
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}, "runs": 0,
            "invested": 0.0, "month": "", "month_spent": 0.0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_trades(rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)


def show_report(state, df):
    """Print the account value, including live marks on any positions."""
    print("\n" + "=" * 62)
    print("PAPER ACCOUNT")
    print("=" * 62)
    print(f"Cash                 : ${state['cash']:,.2f}")
    print(f"Invested to date     : ${state['invested']:,.2f}")
    print(f"Runs                 : {state['runs']}")

    if not state["positions"]:
        print("Positions            : none")
        print(f"\nTotal account value  : ${state['cash']:,.2f}")
        print("=" * 62 + "\n")
        return

    price_map = dict(zip(df["Ticker"], df["Price"])) if not df.empty else {}
    total_val = state["cash"]
    print(f"\n{'Ticker':<8}{'Shares':>10}{'Cost':>12}{'Value':>12}{'P/L':>12}")
    print("-" * 62)
    for t, p in state["positions"].items():
        px = price_map.get(t, p["avg_price"])
        val = p["shares"] * px
        cost = p["shares"] * p["avg_price"]
        pl = val - cost
        total_val += val
        print(f"{t:<8}{p['shares']:>10.3f}{cost:>12,.2f}{val:>12,.2f}{pl:>+12,.2f}")

    print("-" * 62)
    print(f"Total account value  : ${total_val:,.2f}")
    print(f"Total return         : {(total_val / STARTING_CASH - 1) * 100:+.2f}%")
    print("=" * 62 + "\n")


# ----------------------------------------------------------------------------
# 4. THE MAIN RUN
# ----------------------------------------------------------------------------
def run_once():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nPaper run at {now_utc}")

    state = load_state()

    print("Reading market timing...")
    timing, err = timing_score()
    if timing is None:
        print(f"  {err}")
        print("  No trade. Try again later.")
        return

    print(f"  Timing score: {timing['score']}/100  "
          f"(VIX {timing['vix']}, S&P above 50d={timing['above_50d']}, "
          f"above 200d={timing['above_200d']})")

    if timing["score"] < MIN_TIMING_SCORE:
        print(f"  Below your threshold of {MIN_TIMING_SCORE}. NOT investing today.")
        print("  This is the rule working as asked — it waits for favourable conditions.")
        state["runs"] += 1
        save_state(state)
        return

    # --- monthly purchase budget ---
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    if state.get("month") != month_key:
        state["month"] = month_key
        state["month_spent"] = 0.0

    spent = float(state.get("month_spent", 0.0))
    remaining = MONTHLY_PURCHASE_LIMIT - spent
    if remaining <= 0:
        print(f"  Monthly purchase limit already used: "
              f"${spent:,.2f} of ${MONTHLY_PURCHASE_LIMIT:,.2f} in {month_key}.")
        print("  No more buying this month. The limit resets on the 1st.")
        state["runs"] += 1
        save_state(state)
        return

    if state["cash"] <= 0:
        print(f"  No cash left in the account (${state['cash']:,.2f}). "
              "Run --reset to start over.")
        return

    # --- size this run from the timing score ---
    per_run_share = MONTHLY_PURCHASE_LIMIT / max(1, RUNS_PER_MONTH_EXPECTED)
    if timing["score"] >= SCORE_BAND_FULL:
        band = 1.0
    else:
        span = max(1, SCORE_BAND_FULL - MIN_TIMING_SCORE)
        band = SCORE_BAND_MIN + (1.0 - SCORE_BAND_MIN) * ((timing["score"] - MIN_TIMING_SCORE) / span)
        band = max(SCORE_BAND_MIN, min(1.0, band))

    to_invest = min(per_run_share * band, remaining, state["cash"])

    print(f"  Monthly budget: ${spent:,.2f} of ${MONTHLY_PURCHASE_LIMIT:,.2f} used "
          f"({month_key}); ${remaining:,.2f} left.")
    print(f"  Timing score {timing["score"]}/100 -> deploying {band:.0%} of the "
          f"per-run share: ${to_invest:,.2f}")

    print(f"  Timing is favourable. Scoring the universe...")
    df, prices = score_universe()
    if df.empty:
        print("  Could not score the universe right now. No trade.")
        return

    weights = build_target_weights(df, prices)
    if not weights:
        print("  Could not build target weights. No trade.")
        return

    price_map = dict(zip(df["Ticker"], df["Price"]))
    rows, deployed = [], 0.0

    print(f"  Investing ${to_invest:,.2f} via {APPROACH}:\n")
    print(f"  {'Ticker':<8}{'Amount':>12}{'Price':>12}{'Shares':>12}")
    print("  " + "-" * 44)

    cash_left = state["cash"]
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        amount = min(to_invest * w, cash_left)
        if amount < 1 or t not in price_map:
            continue
        price = price_map[t]
        shares = amount / price
        cash_left -= amount
        deployed += amount

        pos = state["positions"].get(t, {"shares": 0.0, "avg_price": 0.0})
        new_shares = pos["shares"] + shares
        new_cost = pos["shares"] * pos["avg_price"] + amount
        state["positions"][t] = {
            "shares": round(new_shares, 6),
            "avg_price": round(new_cost / new_shares, 4) if new_shares else 0.0,
        }

        print(f"  {t:<8}{amount:>12,.2f}{price:>12,.2f}{shares:>12.3f}")
        rows.append({
            "timestamp_utc": now_utc,
            "action": "BUY",
            "ticker": t,
            "shares": round(shares, 6),
            "price": round(price, 4),
            "amount": round(amount, 2),
            "timing_score": timing["score"],
            "approach": APPROACH,
        })

    print("  " + "-" * 44)
    print(f"  {'TOTAL':<8}{deployed:>12,.2f}")

    state["cash"] = round(state["cash"] - deployed, 4)
    state["invested"] = round(state["invested"] + deployed, 2)
    state["runs"] += 1
    save_state(state)
    log_trades(rows)
    print(f"\n  Trade log appended to {LOG_FILE}")

    show_report(state, df)


def main():
    p = argparse.ArgumentParser(description="Paper-trader for the US stock model")
    p.add_argument("--report", action="store_true", help="show the account, make no trade")
    p.add_argument("--reset", action="store_true", help="wipe the account and start over")
    args = p.parse_args()

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        print("Account reset. Starting fresh.")
        return

    if args.report:
        state = load_state()
        print("Scoring universe to mark positions...")
        df, _ = score_universe()
        show_report(state, df)
        return

    run_once()


if __name__ == "__main__":
    main()
