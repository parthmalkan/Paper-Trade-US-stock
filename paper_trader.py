"""
paper_trader.py — Paper trading on the dashboard's own model.

WHAT CHANGED FROM THE OLD VERSION
---------------------------------
The trader used to run its OWN universe (23 names), its own raw-momentum
scoring, no quality factor. It was testing a different strategy from the one
the dashboard shows. Now it imports strategy_core.py, which is the SAME engine
the dashboard uses, so the two cannot drift apart.

BUY RULES
---------
1. Score-scaled, no cliff. The old hard 80 gate meant most runs did nothing.
   Now the amount deployed scales with the timing score: nothing at or below
   SCORE_FLOOR, rising to the full share at 100. Position size carries the
   conviction instead of a yes/no switch.
2. At most one buy per MIN_HOURS_BETWEEN_TRADES (24h) per approach.
3. Monthly ceiling of MONTHLY_PURCHASE_LIMIT.
4. Buys whatever the chosen approach outputs — same picks the dashboard shows.

SELL RULES
----------
1. Score-floor exit: sell when a name's Composite Score falls into the bottom
   SELL_SCORE_PERCENTILE of the universe.
2. Minimum hold of MIN_HOLD_MONTHS before a rank-based exit is allowed, so
   gains can reach long-term capital-gains treatment.
3. Cash-hold: proceeds wait for the next qualifying signal rather than
   immediately re-entering, which halves turnover.

ANALYTICS
---------
Every run appends a snapshot (date, value, cash) to the state file. The
summary prints Today / 1-month / YTD profit per approach, with the benchmark
alongside, modelled on a broker account screen.

NOT FINANCIAL ADVICE. A simulation: no orders are placed anywhere.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

import strategy_core as core

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
STARTING_CASH = core.STARTING_CASH
MIN_PRICE = core.MIN_PRICE

MIN_HOURS_BETWEEN_TRADES = core.MIN_HOURS_BETWEEN_TRADES
MONTHLY_PURCHASE_LIMIT = core.MONTHLY_PURCHASE_LIMIT

# Per-run share = MONTHLY_PURCHASE_LIMIT / RUNS_PER_MONTH_EXPECTED.
# At 20 that was $750, and the score-scaled slice almost never reached the
# daily ceiling. At 5 it is $3,000, so a strong reading deploys real size and
# the $5,000 daily cap becomes the limit that actually bites.
RUNS_PER_MONTH_EXPECTED = 5

# Buy sizing: no cliff. Nothing at or below the floor, full share at 100.
SCORE_FLOOR = 50
SIZE_MIN_FRACTION = 0.25

# Daily ceiling. The monthly allowance is the hard cap; this is how much of it
# any single day may use when conditions are strong. Two runs on the same day
# share it -- the second sees what the first already spent.
DAILY_PURCHASE_LIMIT = 5000.0
# Above this timing score the daily ceiling is allowed to apply in full.
# Below it the day deploys its score-scaled share as before.
DAILY_FULL_SCORE = 75

# Sell rules
SELL_SCORE_PERCENTILE = core.SELL_SCORE_PERCENTILE
MIN_HOLD_MONTHS = core.MIN_HOLD_MONTHS
CASH_HOLD_RUNS = 1

APPROACHES = core.APPROACHES
BENCHMARK_TICKER = core.BENCHMARK_TICKER
BENCHMARK_NAME = core.BENCHMARK_NAME

SUMMARY_FILE = "paper_summary.csv"
SNAPSHOT_LIMIT = 400


# ----------------------------------------------------------------------------
# 1. DATA — same universe and scoring as the dashboard
# ----------------------------------------------------------------------------
def universe_tickers():
    """The bundled membership list. No fragile index fetch."""
    tickers = sorted(core.SP500_SET)
    tickers = [t for t in tickers if t][: core.AUTO_UNIVERSE_SIZE]
    return tickers


def fetch_universe_prices():
    """Six months of closes for the universe. One call per name, skipped on error."""
    tickers = universe_tickers()
    frames = {}
    for i in range(0, len(tickers), 40):
        for t in tickers[i:i + 40]:
            try:
                px = yf.Ticker(t).history(period="6mo", interval="1d",
                                          auto_adjust=True)["Close"].dropna()
                if len(px) >= 20 and float(px.iloc[-1]) >= MIN_PRICE:
                    frames[t] = px
            except Exception:
                continue
    print(f"[prices] usable history for {len(frames)} of {len(tickers)} names")
    return pd.DataFrame(frames).dropna(how="all")


def scored_universe():
    """The scored table both apps share."""
    prices = fetch_universe_prices()
    if prices.empty:
        return pd.DataFrame(), prices
    q = core.quality_from_prices(prices)
    df = core.score_universe(prices, q)
    print(f"[score] {len(df)} names scored")
    return df, prices


def benchmark_history(period="1y"):
    try:
        return yf.Ticker(BENCHMARK_TICKER).history(period=period)["Close"].dropna()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 2. ACCOUNT STATE
# ----------------------------------------------------------------------------
def blank_state():
    return {
        "cash": STARTING_CASH,
        "positions": {},
        "runs": 0,
        "invested": 0.0,
        "month": "",
        "month_spent": 0.0,
        "day": "",
        "day_spent": 0.0,
        "last_trade_utc": "",
        "last_trade_score": 0,
        "snapshots": [],
        "hold_since": {},
        "cash_hold": 0,
    }


def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                st = json.load(f)
            for k, v in blank_state().items():
                st.setdefault(k, v)
            return st
        except Exception:
            pass
    return blank_state()


def save_state(state, path):
    if len(state.get("snapshots", [])) > SNAPSHOT_LIMIT:
        state["snapshots"] = state["snapshots"][-SNAPSHOT_LIMIT:]
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def log_trades(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def hours_since(iso_utc):
    if not iso_utc:
        return None
    try:
        t = datetime.strptime(iso_utc, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
    except Exception:
        return None


def months_held(state, ticker):
    since = state.get("hold_since", {}).get(ticker)
    if not since:
        return None
    try:
        d = datetime.strptime(since, "%Y-%m-%d").date()
    except Exception:
        return None
    today = datetime.now(timezone.utc).date()
    return (today.year - d.year) * 12 + (today.month - d.month)


def account_value(state, price_map):
    total = float(state.get("cash", 0.0))
    for t, p in state.get("positions", {}).items():
        total += p.get("shares", 0.0) * float(price_map.get(t, p.get("avg_price", 0.0)))
    return total


# ----------------------------------------------------------------------------
# 3. BUY SIZING — score-scaled, no cliff
# ----------------------------------------------------------------------------
def size_fraction(score):
    """Fraction of the per-run share to deploy at this timing score.

    No hard gate: at SCORE_FLOOR it deploys SIZE_MIN_FRACTION, rising linearly
    to 1.0 at 100. At or below the floor it deploys nothing, because a score
    that low is the rule reading genuinely poor conditions.
    """
    if score <= SCORE_FLOOR:
        return 0.0
    span = max(1, 100 - SCORE_FLOOR)
    frac = SIZE_MIN_FRACTION + (1.0 - SIZE_MIN_FRACTION) * ((score - SCORE_FLOOR) / span)
    return max(SIZE_MIN_FRACTION, min(1.0, frac))


def reset_day_if_needed(state, today):
    """Roll the daily spend counter over on a new calendar day.

    This lives OUT here, not inside daily_budget(), because daily_budget() is
    only reached once every early-return guard has been passed. A run that
    exits on the 24h spacing, the cash-hold, or the monthly ceiling would
    otherwise leave yesterday's day_spent in place, and the next run would
    measure a fresh day's ceiling against stale spend.
    """
    if state.get("day") != today:
        state["day"] = today
        state["day_spent"] = 0.0


def daily_budget(score, state, today):
    """How much this run may deploy today, before the monthly ceiling.

    Three things are in play, smallest wins:

      monthly share  the score-scaled slice of the monthly allowance
      daily ceiling  DAILY_PURCHASE_LIMIT, reached in full once the score is
                     at or above DAILY_FULL_SCORE; scaled below that, so a
                     mediocre day cannot spend the whole daily allowance
      daily left     the daily ceiling minus what earlier runs today spent

    The monthly allowance stays the outer bound: the caller still min()s this
    against what is left of MONTHLY_PURCHASE_LIMIT.

    The caller is responsible for calling reset_day_if_needed() first.
    """
    frac = size_fraction(score)
    if frac <= 0:
        return 0.0

    per_run = MONTHLY_PURCHASE_LIMIT / max(1, RUNS_PER_MONTH_EXPECTED)
    monthly_share = per_run * frac

    # the ceiling only opens fully on a strong reading
    if score >= DAILY_FULL_SCORE:
        ceiling_frac = 1.0
    else:
        span = max(1, DAILY_FULL_SCORE - SCORE_FLOOR)
        ceiling_frac = max(SIZE_MIN_FRACTION,
                           (score - SCORE_FLOOR) / span)
    ceiling = DAILY_PURCHASE_LIMIT * min(1.0, ceiling_frac)

    day_left = max(0.0, ceiling - float(state.get("day_spent", 0.0)))
    return min(monthly_share, ceiling, day_left)


# ----------------------------------------------------------------------------
# 4. SELL RULES
# ----------------------------------------------------------------------------
def sell_decisions(state, df):
    """Which held names should exit, and why.

    Rule: Composite Score in the bottom SELL_SCORE_PERCENTILE of the universe.
    Guard: skip any name held less than MIN_HOLD_MONTHS so gains can reach
    long-term treatment.
    """
    if df is None or df.empty:
        return []
    scores = dict(zip(df["Ticker"], df["Composite Score"]))
    prices = dict(zip(df["Ticker"], df["Price"]))
    cutoff = float(df["Composite Score"].quantile(SELL_SCORE_PERCENTILE))

    out = []
    for t, pos in state.get("positions", {}).items():
        held = months_held(state, t)
        if held is None or held < MIN_HOLD_MONTHS:
            continue
        s = scores.get(t)
        if s is None:
            continue
        if s < cutoff:
            out.append({
                "ticker": t,
                "reason": f"score {s:.0f} below the bottom-third floor ({cutoff:.0f})",
                "price": float(prices.get(t, pos.get("avg_price", 0.0))),
            })
    return out


# ----------------------------------------------------------------------------
# 5. RUN ONE APPROACH
# ----------------------------------------------------------------------------
def run_approach(approach, state_file, log_file, timing, df, prices):
    """Apply every rule for ONE approach, on its own account."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state = load_state(state_file)
    state["runs"] = state.get("runs", 0) + 1
    msgs = []

    # Roll the day and month counters over BEFORE any guard can return, so a
    # run that exits early still leaves a clean slate for the next one.
    reset_day_if_needed(state, today)
    month_key = today[:7]
    if state.get("month") != month_key:
        state["month"] = month_key
        state["month_spent"] = 0.0

    score = int(timing["score"]) if timing else 50
    price_map = dict(zip(df["Ticker"], df["Price"])) if not df.empty else {}

    # ---------- SELLS first, so freed cash can be redeployed ----------
    for d in sell_decisions(state, df):
        t = d["ticker"]
        pos = state["positions"].get(t)
        if not pos:
            continue
        proceeds = pos["shares"] * d["price"]
        state["cash"] = round(state["cash"] + proceeds, 2)
        del state["positions"][t]
        state.get("hold_since", {}).pop(t, None)
        state["cash_hold"] = CASH_HOLD_RUNS
        log_trades([{
            "timestamp_utc": now_utc, "action": "SELL", "ticker": t,
            "shares": round(pos["shares"], 6), "price": round(d["price"], 4),
            "amount": round(proceeds, 2), "timing_score": score,
            "approach": approach, "reason": d["reason"],
        }], log_file)
        msgs.append(f"sold {t} ({d['reason']})")

    # ---------- BUY GATE ----------
    frac = size_fraction(score)
    if frac <= 0:
        save_state(state, state_file)
        msgs.append(f"score {score} at/below floor {SCORE_FLOOR} - no buy")
        return approach, msgs

    elapsed = hours_since(state.get("last_trade_utc", ""))
    if elapsed is not None and elapsed < MIN_HOURS_BETWEEN_TRADES:
        save_state(state, state_file)
        msgs.append(f"only {elapsed:.1f}h since last trade - no buy")
        return approach, msgs

    if float(state.get("cash_hold", 0)) > 0:
        state["cash_hold"] = max(0, int(state["cash_hold"]) - 1)
        save_state(state, state_file)
        msgs.append("cash-hold in effect - proceeds waiting for the next signal")
        return approach, msgs

    spent = float(state.get("month_spent", 0.0))
    remaining = MONTHLY_PURCHASE_LIMIT - spent
    if remaining <= 0:
        save_state(state, state_file)
        msgs.append(f"monthly limit used (${spent:,.2f}) - no buy")
        return approach, msgs

    # Score-scaled slice of the month, capped by today's ceiling and by what
    # earlier runs today already spent. The monthly allowance is the outer bound.
    day_allowance = daily_budget(score, state, today)
    to_invest = min(day_allowance, remaining, float(state["cash"]))
    if to_invest < 1:
        save_state(state, state_file)
        if day_allowance <= 0:
            msgs.append(f"today's ceiling reached "
                        f"(${float(state.get('day_spent', 0.0)):,.2f} of "
                        f"${DAILY_PURCHASE_LIMIT:,.2f}) - no buy")
        else:
            msgs.append("nothing investable at this size")
        return approach, msgs

    # ---------- PICKS: the dashboard's own output ----------
    picks = core.picks_for(approach, df, prices)
    if picks is None or picks.empty:
        save_state(state, state_file)
        msgs.append("no picks returned by the model")
        return approach, msgs

    rows, deployed = [], 0.0
    cash_left = float(state["cash"])
    for _, r in picks.iterrows():
        t = r["Ticker"]
        w = float(r.get("Weight", 0.0))
        amount = min(to_invest * w, cash_left)
        if amount < 1 or t not in price_map:
            continue
        price = float(price_map[t])
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
        state.setdefault("hold_since", {}).setdefault(t, today)
        rows.append({
            "timestamp_utc": now_utc, "action": "BUY", "ticker": t,
            "shares": round(shares, 6), "price": round(price, 4),
            "amount": round(amount, 2), "timing_score": score,
            "approach": approach, "reason": "",
        })

    if deployed <= 0:
        save_state(state, state_file)
        msgs.append("no names investable")
        return approach, msgs

    state["cash"] = round(state["cash"] - deployed, 2)
    state["invested"] = round(state.get("invested", 0.0) + deployed, 2)
    state["month_spent"] = round(spent + deployed, 2)
    state["day_spent"] = round(float(state.get("day_spent", 0.0)) + deployed, 2)
    state["last_trade_utc"] = now_utc
    state["last_trade_score"] = score
    log_trades(rows, log_file)

    # ---------- SNAPSHOT for the analytics ----------
    state.setdefault("snapshots", []).append({
        "date": today,
        "value": round(account_value(state, price_map), 2),
        "cash": round(state["cash"], 2),
    })
    save_state(state, state_file)
    msgs.append(f"bought ${deployed:,.2f} across {len(rows)} names at score {score}")
    return approach, msgs


# ----------------------------------------------------------------------------
# 6. ANALYTICS — Today / 1 month / YTD, broker-style
# ----------------------------------------------------------------------------
def period_change(snaps, value_now, days):
    """Account-value change over the last N calendar days, from snapshots."""
    if not snaps:
        return None
    today = datetime.now(timezone.utc).date()
    target = today - pd.Timedelta(days=days)
    target = target.date() if hasattr(target, "date") else target
    past = None
    for s in snaps:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if d <= target:
            past = s
    if past is None:
        return None
    return value_now - float(past.get("value", value_now))


def ytd_change(snaps, value_now):
    if not snaps:
        return None
    year = str(datetime.now(timezone.utc).year)
    for s in snaps:
        if str(s.get("date", "")).startswith(year):
            return value_now - float(s.get("value", value_now))
    return None


def write_summary():
    """Per-approach analytics, modelled on a broker account screen."""
    df, _prices = scored_universe()
    price_map = dict(zip(df["Ticker"], df["Price"])) if not df.empty else {}

    bench = benchmark_history("1y")
    bench_1m = bench_ytd = None
    if bench is not None and len(bench) > 2:
        try:
            b_now = float(bench.iloc[-1])
            b_1m = float(bench.iloc[-21]) if len(bench) > 21 else b_now
            bench_1m = (b_now / b_1m - 1) * 100
            year = str(datetime.now(timezone.utc).year)
            ys = bench[bench.index >= f"{year}-01-01"]
            if len(ys):
                bench_ytd = (b_now / float(ys.iloc[0]) - 1) * 100
        except Exception:
            pass

    rows = []
    for approach, state_file, log_file in APPROACHES:
        st = load_state(state_file)
        val = account_value(st, price_map)
        snaps = st.get("snapshots", [])

        today_pl = period_change(snaps, val, 1)
        month_pl = period_change(snaps, val, 30)
        ytd_pl = ytd_change(snaps, val)

        trades = 0
        if os.path.exists(log_file):
            try:
                trades = len(pd.read_csv(log_file))
            except Exception:
                trades = 0

        def pct(x):
            return round(x / STARTING_CASH * 100, 2) if x is not None else None

        rows.append({
            "Approach": approach,
            "Total value": round(val, 2),
            "Total return %": round((val / STARTING_CASH - 1) * 100, 2),
            "Today P/L": round(today_pl, 2) if today_pl is not None else None,
            "1-month P/L": round(month_pl, 2) if month_pl is not None else None,
            "YTD P/L": round(ytd_pl, 2) if ytd_pl is not None else None,
            "Today %": pct(today_pl),
            "1-month %": pct(month_pl),
            "YTD %": pct(ytd_pl),
            "Cash": round(st.get("cash", 0.0), 2),
            "Positions": len(st.get("positions", {})),
            "Trades": trades,
        })

    summary = pd.DataFrame(rows).sort_values("Total return %", ascending=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    def money(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"${v:+,.2f}"

    def pcts(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:+.2f}%"

    print("\n" + "=" * 78)
    print("PAPER ACCOUNTS  -  broker-style summary")
    print("=" * 78)
    for _, r in summary.iterrows():
        print(f"\n  {r['Approach']}")
        print(f"    Value ${r['Total value']:,.2f}   total {r['Total return %']:+.2f}%")
        print(f"    Today    {money(r['Today P/L']):>12}   {pcts(r['Today %']):>8}")
        print(f"    1 month  {money(r['1-month P/L']):>12}   {pcts(r['1-month %']):>8}")
        print(f"    YTD      {money(r['YTD P/L']):>12}   {pcts(r['YTD %']):>8}")
        print(f"    Cash ${r['Cash']:,.2f}   positions {r['Positions']}   trades {r['Trades']}")

    if bench_1m is not None:
        print(f"\n  {BENCHMARK_NAME}: 1 month {bench_1m:+.2f}%  YTD {bench_ytd:+.2f}%")
    else:
        print(f"\n  {BENCHMARK_NAME}: benchmark unavailable")

    best = summary.iloc[0] if len(summary) else None
    if best is not None:
        print(f"\n  Leader: {best['Approach']} at {best['Total return %']:+.2f}%.")
        print("  A short history is not evidence. Give all three the same stretch")
        print("  of market before drawing conclusions.")
    print("=" * 78)
    return summary


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Paper trader on the dashboard model")
    p.add_argument("--report", action="store_true", help="summarise, make no trade")
    p.add_argument("--reset", action="store_true", help="wipe all accounts")
    args = p.parse_args()

    if args.reset:
        for _, sf, lf in APPROACHES:
            for path in (sf, lf):
                if os.path.exists(path):
                    os.remove(path)
        if os.path.exists(SUMMARY_FILE):
            os.remove(SUMMARY_FILE)
        print("All three accounts reset.")
        return

    if args.report:
        write_summary()
        return

    timing = core.market_timing()
    if timing is None:
        print("Could not read market indicators. No trade.")
        return
    print(f"Timing score {timing['score']}/100 ({timing['regime']})  "
          f"VIX {timing['vix']}  S&P {timing['sp500']}")

    df, prices = scored_universe()
    if df.empty:
        print("Could not score the universe. No trade.")
        return

    for approach, sf, lf in APPROACHES:
        a, msgs = run_approach(approach, sf, lf, timing, df, prices)
        for m in msgs:
            print(f"  {a}: {m}")

    write_summary()


if __name__ == "__main__":
    main()
