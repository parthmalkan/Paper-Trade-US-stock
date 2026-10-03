"""
paper_view.py — Read-only Streamlit screen for the three paper-trading accounts.

WHAT IT IS
----------
The display half of the paper trader. It reads the files the trader produces
and renders them live in a browser: one tab per approach, plus a summary that
ranks them.

It never trades and never writes. paper_trader.py does the trading, on GitHub,
on a schedule.

FILES IT READS
--------------
  paper_account_a1.json / paper_trades_a1.csv   -> Approach 1 (diversified)
  paper_account_a2.json / paper_trades_a2.csv   -> Approach 2 (momentum)
  paper_account_a3.json / paper_trades_a3.csv   -> Approach 3 (optimizer)
  paper_summary.csv                              -> written by the trader

The Today / 1-month / YTD figures come from the `snapshots` array the trader
appends to each account on every run that trades. On a fresh account they read
"pending" until a few snapshots have accumulated -- that is expected, not a bug.

DEPLOY
------
1. Put this file, paper_trader.py, strategy_core.py, requirements.txt and the
   account files in one GitHub repo.
2. share.streamlit.io -> New app -> pick the repo -> main file: paper_view.py
3. Set REPO_RAW_BASE below so the page reads the newest committed data.

Settings are imported from strategy_core where possible, so the numbers shown
here cannot drift from the numbers the trader actually uses.

Not financial advice. A simulation: no orders are placed anywhere.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS — read from strategy_core so the two cannot disagree
# ----------------------------------------------------------------------------
try:
    import strategy_core as core
    STARTING_CASH = core.STARTING_CASH
    MONTHLY_PURCHASE_LIMIT = core.MONTHLY_PURCHASE_LIMIT
    MIN_HOURS_BETWEEN_TRADES = core.MIN_HOURS_BETWEEN_TRADES
    SELL_SCORE_PERCENTILE = core.SELL_SCORE_PERCENTILE
    MIN_HOLD_MONTHS = core.MIN_HOLD_MONTHS
    BENCHMARK_NAME = core.BENCHMARK_NAME
    BENCHMARK_TICKER = core.BENCHMARK_TICKER
except Exception:
    # same values as strategy_core, so a missing import degrades to a correct
    # page rather than a broken one
    STARTING_CASH = 100000.0
    MONTHLY_PURCHASE_LIMIT = 15000.0
    MIN_HOURS_BETWEEN_TRADES = 24
    SELL_SCORE_PERCENTILE = 0.33
    MIN_HOLD_MONTHS = 3
    BENCHMARK_NAME = "S&P 500"
    BENCHMARK_TICKER = "^GSPC"

# these two live only in paper_trader.py, so they are mirrored here
SCORE_FLOOR = 50
DAILY_PURCHASE_LIMIT = 5000.0
RUNS_PER_MONTH_EXPECTED = 5

APPROACHES = [
    ("Approach 1", "Diversified large-cap",
     "paper_account_a1.json", "paper_trades_a1.csv"),
    ("Approach 2", "Momentum / speculative",
     "paper_account_a2.json", "paper_trades_a2.csv"),
    ("Approach 3", "Markowitz optimizer",
     "paper_account_a3.json", "paper_trades_a3.csv"),
]

# Set this to your repo's raw URL so the page always reads the newest commit.
# e.g. "https://raw.githubusercontent.com/<user>/<repo>/main"
# Leave blank to read the local copies that came with the deploy.
REPO_RAW_BASE = ""

st.set_page_config(page_title="Paper Trading Accounts", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------------
def _fetch(name):
    """Text of a file, from the repo if configured, else from disk."""
    if REPO_RAW_BASE:
        try:
            import urllib.request
            stamp = int(datetime.now().timestamp())
            url = f"{REPO_RAW_BASE.rstrip('/')}/{name}?t={stamp}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode("utf-8")
        except Exception:
            pass
    if os.path.exists(name):
        try:
            with open(name, encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None


def load_json(name):
    txt = _fetch(name)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def load_csv(name):
    txt = _fetch(name)
    if not txt:
        return None
    try:
        from io import StringIO
        return pd.read_csv(StringIO(txt))
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def live_prices(tickers):
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          threads=True, progress=False)
        out = {}
        for t in tickers:
            try:
                px = raw[t]["Close"].dropna()
                if len(px):
                    out[t] = float(px.iloc[-1])
            except Exception:
                continue
        return out
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def timing_now():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        spx = yf.Ticker("^GSPC").history(period="260d")["Close"]
        now = spx.iloc[-1]
        ma50 = spx.rolling(50).mean().iloc[-1]
        ma200 = spx.rolling(200).mean().iloc[-1]
    except Exception:
        return None
    score = 50
    score += 20 if (now > ma50 and now > ma200) else -20
    score += 15 if vix < 20 else (0 if vix < 30 else -15)
    score = max(0, min(100, score))
    return {"score": int(score), "vix": round(float(vix), 2),
            "sp500": round(float(now), 2),
            "above_50d": bool(now > ma50), "above_200d": bool(now > ma200)}


@st.cache_data(ttl=300, show_spinner=False)
def benchmark_windows():
    """Benchmark change over 1 day, 30 days and year-to-date."""
    try:
        px = yf.Ticker(BENCHMARK_TICKER).history(period="1y")["Close"].dropna()
    except Exception:
        return None
    if len(px) < 2:
        return None
    try:
        now = float(px.iloc[-1])
        out = {"now": now}
        out["d1"] = (now / float(px.iloc[-2]) - 1) * 100 if len(px) > 2 else 0.0
        out["d30"] = (now / float(px.iloc[-21]) - 1) * 100 if len(px) > 21 else None
        year = str(datetime.now(timezone.utc).year)
        ys = px[px.index >= f"{year}-01-01"]
        out["ytd"] = (now / float(ys.iloc[0]) - 1) * 100 if len(ys) else None
        return out
    except Exception:
        return None


def period_change(snaps, value_now, days):
    """Account-value change over the last N calendar days, from snapshots.

    Returns None when no snapshot is old enough yet -- the caller shows
    "pending" rather than inventing a number.
    """
    if not snaps:
        return None
    target = datetime.now(timezone.utc).date() - pd.Timedelta(days=days)
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


def account_snapshot(state, prices):
    """Value, return, and position rows for one account."""
    positions = state.get("positions", {}) if state else {}
    cash = float(state.get("cash", 0.0)) if state else 0.0
    rows, total = [], cash
    for tk, p in sorted(positions.items()):
        px = float(prices.get(tk, p.get("avg_price", 0.0)))
        shares = p.get("shares", 0.0)
        cost = shares * p.get("avg_price", 0.0)
        val = shares * px
        pl = val - cost
        total += val
        rows.append({
            "Ticker": tk,
            "Shares": round(shares, 3),
            "Avg cost": round(p.get("avg_price", 0.0), 2),
            "Price": round(px, 2),
            "Cost basis": round(cost, 2),
            "Value": round(val, 2),
            "P/L ($)": round(pl, 2),
            "P/L (%)": round((pl / cost * 100) if cost else 0.0, 1),
        })
    ret = (total / STARTING_CASH - 1) * 100 if STARTING_CASH else 0.0
    return total, ret, cash, rows, positions


def colour_pl(v):
    if isinstance(v, (int, float)):
        if v >= 0:
            return "color: #1a7f37; font-weight: 600"
        return "color: #cf222e; font-weight: 600"
    return ""


def money_or_pending(v, snaps_available):
    if v is None:
        return "pending" if snaps_available else "—"
    return f"${v:+,.2f}"


def pct_or_pending(v, snaps_available):
    if v is None:
        return "pending" if snaps_available else "—"
    return f"{v:+.2f}%"


# ----------------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------------
st.title("📄 Paper Trading Accounts")
st.caption("Three approaches, three separate accounts. Read-only — the GitHub "
           "Action does the trading on its schedule.")

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

timing = timing_now()
if timing:
    above = timing["score"] > SCORE_FLOOR
    gate = "trading enabled" if above else "below the floor — no buys"
    colour = "green" if above else "orange"
    st.markdown(
        f"**Market timing: :{colour}[{timing['score']}/100]** — {gate}. "
        f"VIX {timing['vix']} · S&P 500 {timing['sp500']:,} · "
        f"above 50d: {'yes' if timing['above_50d'] else 'no'} · "
        f"above 200d: {'yes' if timing['above_200d'] else 'no'}"
    )
    st.progress(min(1.0, timing["score"] / 100))
    st.caption(
        f"Below {SCORE_FLOOR} the trader deploys nothing. Above it, size scales "
        f"with the score — about {DAILY_PURCHASE_LIMIT:,.0f} per day at most, "
        f"out of the monthly {MONTHLY_PURCHASE_LIMIT:,.0f} allowance."
    )
else:
    st.info("Could not read the market indicators right now.")

st.divider()

bench = benchmark_windows()

# Load every account once
loaded = []
for name, blurb, state_file, log_file in APPROACHES:
    stt = load_json(state_file)
    trd = load_csv(log_file)
    loaded.append({"name": name, "blurb": blurb, "state": stt, "trades": trd})

all_tickers = sorted({t for a in loaded
                      for t in (a["state"].get("positions", {}) if a["state"] else {})})
prices = live_prices(tuple(all_tickers))

# ---------------- analytics strip: Today / 1 month / YTD ----------------
st.subheader("📊 Performance")
st.caption("Modelled on a broker account screen. Figures come from the "
           "snapshots the trader records on every run that trades.")

perf_rows = []
for a in loaded:
    stt = a["state"]
    if stt is None:
        perf_rows.append({
            "Approach": a["name"], "Value": None, "Today": None,
            "1 month": None, "YTD": None, "Total": None,
        })
        continue
    total, ret, _cash, _rows, _pos = account_snapshot(stt, prices)
    snaps = stt.get("snapshots", [])
    today_pl = period_change(snaps, total, 1)
    month_pl = period_change(snaps, total, 30)
    ytd_pl = ytd_change(snaps, total)

    def as_pct(x):
        return round(x / STARTING_CASH * 100, 2) if x is not None else None

    perf_rows.append({
        "Approach": a["name"],
        "Value": round(total, 2),
        "Today": as_pct(today_pl),
        "1 month": as_pct(month_pl),
        "YTD": as_pct(ytd_pl),
        "Total": round(ret, 2),
    })

pdf = pd.DataFrame(perf_rows)

if bench:
    bench_row = {
        "Approach": f"{BENCHMARK_NAME} (benchmark)",
        "Value": None,
        "Today": round(bench["d1"], 2) if bench.get("d1") is not None else None,
        "1 month": round(bench["d30"], 2) if bench.get("d30") is not None else None,
        "YTD": round(bench["ytd"], 2) if bench.get("ytd") is not None else None,
        "Total": None,
    }
    pdf_display = pd.concat([pdf, pd.DataFrame([bench_row])], ignore_index=True)
else:
    pdf_display = pdf

st.dataframe(
    pdf_display.style.format({
        "Value": "${:,.2f}",
        "Today": "{:+.2f}%",
        "1 month": "{:+.2f}%",
        "YTD": "{:+.2f}%",
        "Total": "{:+.2f}%",
    }, na_rep="pending").map(colour_pl, subset=["Today", "1 month", "YTD", "Total"]),
    use_container_width=True, hide_index=True,
)

any_snaps = any((a["state"] or {}).get("snapshots") for a in loaded)
if not any_snaps:
    st.info(
        "The period columns read **pending** until the trader has traded a few "
        "times. Each run that trades records a dated snapshot, and Today / "
        "1-month / YTD are measured against those. Nothing is wrong — there is "
        "simply no history yet."
    )

st.divider()

tab_names = [a["name"] for a in loaded] + ["🏆 Summary"]
tabs = st.tabs(tab_names)

# ---------------- per-approach tabs ----------------
for i, a in enumerate(loaded):
    with tabs[i]:
        st.subheader(a["name"] + " — " + a["blurb"])
        if a["state"] is None:
            st.warning("No account yet. `" + APPROACHES[i][2] + "` is not in the "
                       "repo, or the trader has not run for this approach.")
            continue

        total, ret, cash, rows, positions = account_snapshot(a["state"], prices)
        stt = a["state"]
        snaps = stt.get("snapshots", [])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total value", f"${total:,.2f}", f"{ret:+.2f}%")
        c2.metric("Cash", f"${cash:,.2f}")
        c3.metric("Positions", len(positions))
        c4.metric("Runs", stt.get("runs", 0))

        # --- period P/L, broker style ---
        today_pl = period_change(snaps, total, 1)
        month_pl = period_change(snaps, total, 30)
        ytd_pl = ytd_change(snaps, total)

        def pct_of_cash(x):
            return round(x / STARTING_CASH * 100, 2) if x is not None else None

        p1, p2, p3 = st.columns(3)
        p1.metric("Today P/L", money_or_pending(today_pl, bool(snaps)),
                  pct_or_pending(pct_of_cash(today_pl), bool(snaps)))
        p2.metric("1-month P/L", money_or_pending(month_pl, bool(snaps)),
                  pct_or_pending(pct_of_cash(month_pl), bool(snaps)))
        p3.metric("YTD P/L", money_or_pending(ytd_pl, bool(snaps)),
                  pct_or_pending(pct_of_cash(ytd_pl), bool(snaps)))

        # --- budget ---
        spent = float(stt.get("month_spent", 0.0))
        day_spent = float(stt.get("day_spent", 0.0))
        mkey = stt.get("month", "") or "n/a"
        st.markdown("**Budget — " + mkey + "**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Spent this month", f"${spent:,.0f}")
        b2.metric("Remaining this month",
                  f"${max(0, MONTHLY_PURCHASE_LIMIT - spent):,.0f}")
        b3.metric("Spent today", f"${day_spent:,.0f}",
                  f"of ${DAILY_PURCHASE_LIMIT:,.0f}")
        _pct = (spent / MONTHLY_PURCHASE_LIMIT) if MONTHLY_PURCHASE_LIMIT else 0
        st.progress(min(1.0, _pct))
        st.caption(
            f"{_pct:.0%} of the ${MONTHLY_PURCHASE_LIMIT:,.0f} monthly "
            f"allowance used. The daily ceiling is ${DAILY_PURCHASE_LIMIT:,.0f}, "
            f"reached in full once the timing score is strong."
        )

        last = stt.get("last_trade_utc", "") or "never"
        last_score = stt.get("last_trade_score", 0)
        _line = "Last trade: **" + last + "**"
        if last != "never":
            _line += f" at score {last_score}"
        _line += (f" · Rules: max 1 buy / {MIN_HOURS_BETWEEN_TRADES}h · "
                  f"sells only after {MIN_HOLD_MONTHS} months, when the score "
                  f"falls into the bottom {SELL_SCORE_PERCENTILE:.0%}.")
        st.caption(_line)

        st.markdown("**Positions**")
        if not rows:
            st.info("No positions yet — this approach has not traded.")
        else:
            dfp = pd.DataFrame(rows).sort_values("Value", ascending=False)
            hold = stt.get("hold_since", {})
            if hold:
                dfp["Held since"] = dfp["Ticker"].map(lambda t: hold.get(t, "—"))
            fmt = {
                "Shares": "{:,.3f}", "Avg cost": "${:,.2f}", "Price": "${:,.2f}",
                "Cost basis": "${:,.2f}", "Value": "${:,.2f}",
                "P/L ($)": "${:+,.2f}", "P/L (%)": "{:+.1f}%",
            }
            st.dataframe(
                dfp.style.format(fmt).map(colour_pl, subset=["P/L ($)", "P/L (%)"]),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Trade history**")
        if a["trades"] is None or a["trades"].empty:
            st.info("No trades logged yet.")
        else:
            sh = a["trades"].iloc[::-1]
            cols = [c for c in ["timestamp_utc", "action", "ticker", "shares",
                                "price", "amount", "timing_score", "reason"]
                    if c in sh.columns]
            sh = sh[cols].rename(columns={
                "timestamp_utc": "When (UTC)", "action": "Side", "ticker": "Ticker",
                "shares": "Shares", "price": "Price", "amount": "Amount",
                "timing_score": "Timing", "reason": "Reason"})
            fmt = {"Shares": "{:,.3f}", "Price": "${:,.2f}",
                   "Amount": "${:,.2f}", "Timing": "{:.0f}"}
            st.dataframe(sh.style.format(fmt), use_container_width=True,
                         hide_index=True)
            st.caption(f"{len(sh)} trades.")

# ---------------- summary tab ----------------
with tabs[len(loaded)]:
    st.subheader("🏆 Which approach is best so far?")
    st.caption("Ranked by total return, which blends what each approach bought "
               "with the cash it left idle.")

    comp = []
    for a in loaded:
        stt = a["state"]
        if stt is None:
            comp.append({"Approach": a["name"], "Total value": None,
                         "Return (%)": None, "Cash": None, "Positions": 0,
                         "Trades": 0, "Last trade": "-"})
            continue
        total, ret, cash, rows, positions = account_snapshot(stt, prices)
        ntrades = 0 if a["trades"] is None or a["trades"].empty else len(a["trades"])
        comp.append({
            "Approach": a["name"],
            "Total value": round(total, 2),
            "Return (%)": round(ret, 2),
            "Cash": round(cash, 2),
            "Positions": len(positions),
            "Trades": ntrades,
            "Last trade": stt.get("last_trade_utc", "") or "never",
        })

    cdf = pd.DataFrame(comp)
    ranked = cdf.dropna(subset=["Return (%)"]).sort_values("Return (%)",
                                                           ascending=False)

    if ranked.empty:
        st.info("No account has traded yet, so there is nothing to rank. All "
                f"{len(loaded)} approaches are waiting for the timing score to "
                f"clear {SCORE_FLOOR}.")
    else:
        best = ranked.iloc[0]
        st.markdown(f"### Leader: **{best['Approach']}** at "
                    f"{best['Return (%)']:+.2f}%")

        st.dataframe(
            cdf.style.format({
                "Total value": "${:,.2f}", "Return (%)": "{:+.2f}%",
                "Cash": "${:,.2f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        if bench and bench.get("ytd") is not None:
            st.markdown("**Against the benchmark**")
            for _, r in ranked.iterrows():
                pass
            b1, b2, b3 = st.columns(3)
            b1.metric(f"{BENCHMARK_NAME} today",
                      f"{bench['d1']:+.2f}%" if bench.get("d1") is not None else "—")
            b2.metric(f"{BENCHMARK_NAME} 1 month",
                      f"{bench['d30']:+.2f}%" if bench.get("d30") is not None else "—")
            b3.metric(f"{BENCHMARK_NAME} YTD",
                      f"{bench['ytd']:+.2f}%")

        st.markdown("**Why this ranking**")
        st.markdown(
            f"- **{best['Approach']}** leads on total return, which counts both "
            "the positions it holds and the cash it kept uninvested. An approach "
            "that trades often but holds cash earns less on that idle money.\n"
            f"- A high cash balance usually means the timing score sat below "
            f"{SCORE_FLOOR} and the trader stayed out — disciplined in a falling "
            "market, costly in a rising one.\n"
            "- Trade count matters twice: every trade is a decision, and the "
            "fewer trades needed for the same return, the less the approach "
            "relied on activity to get there.\n"
            "- A short history is not evidence. Give all three approaches the "
            "same stretch of market before drawing conclusions."
        )

        st.caption("Rankings come from the committed account files. If an "
                   "approach shows no trades, that is the timing floor keeping "
                   "it out, not an error.")

st.divider()
st.caption("Simulation only — no real orders were placed and no brokerage "
           "account is connected. Prices come from Yahoo's free feed and may be "
           "delayed. Decision-support tool, not financial advice.")
