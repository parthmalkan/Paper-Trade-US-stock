"""
paper_view.py — Read-only Streamlit screen for the paper trading account.

WHAT IT IS
----------
The display half of the paper trader. It reads the files the trader produces
(paper_account.json and paper_trades.csv) and renders them live in a browser:
account value, positions with current P/L, the trade history, the timing score
and the monthly budget used.

It never trades and never writes. paper_trader.py does the trading.

DEPLOY (Streamlit Community Cloud)
----------------------------------
1. Put this file, paper_trader.py, requirements.txt, paper_account.json and
   paper_trades.csv in one GitHub repo.
2. share.streamlit.io -> New app -> pick the repo -> main file: paper_view.py
3. Open the URL any time, on any device. No downloads.

The Refresh button re-reads the files from GitHub, so you see the latest run
without waiting for a cache to expire.
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
# SETTINGS — keep these in step with paper_trader.py
# ----------------------------------------------------------------------------
STATE_FILE = "paper_account.json"
LOG_FILE = "paper_trades.csv"
STARTING_CASH = 100000.0
MONTHLY_PURCHASE_LIMIT = 15000.0
MIN_TIMING_SCORE = 80

# On Streamlit Cloud the repo is cloned at deploy time, so the files on disk can
# be stale. Set your repo here (owner/name) to pull the newest version instead.
# Leave blank to just read the local copies.
REPO_RAW_BASE = "[raw.githubusercontent.com](https://raw.githubusercontent.com/parthmalkan/Paper-Trade-US-stock/main)"      # e.g. "https://raw.githubusercontent.com/yourname/stock-model/main"

st.set_page_config(page_title="Paper Trading Account", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------------
def load_json(name):
    """Read a JSON file, from the repo if configured, else from disk."""
    if REPO_RAW_BASE:
        try:
            import urllib.request
            url = f"{REPO_RAW_BASE.rstrip('/')}/{name}?t={int(datetime.now().timestamp())}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    if os.path.exists(name):
        try:
            with open(name) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_csv(name):
    """Read the trade log, from the repo if configured, else from disk."""
    if REPO_RAW_BASE:
        try:
            import urllib.request
            url = f"{REPO_RAW_BASE.rstrip('/')}/{name}?t={int(datetime.now().timestamp())}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                from io import StringIO
                return pd.read_csv(StringIO(r.read().decode("utf-8")))
        except Exception:
            pass
    if os.path.exists(name):
        try:
            return pd.read_csv(name)
        except Exception:
            return None
    return None


@st.cache_data(ttl=300, show_spinner=False)
def live_prices(tickers):
    """Current prices for the held tickers, cached for 5 minutes."""
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
    """Same timing rule as the trader, so the screen agrees with the log."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        spx = yf.Ticker("^GSPC").history(period="260d")["Close"]
        now, ma50, ma200 = spx.iloc[-1], spx.rolling(50).mean().iloc[-1], spx.rolling(200).mean().iloc[-1]
    except Exception:
        return None
    score = 50
    score += 20 if (now > ma50 and now > ma200) else -20
    score += 15 if vix < 20 else (0 if vix < 30 else -15)
    score = max(0, min(100, score))
    return {"score": int(score), "vix": round(float(vix), 2),
            "sp500": round(float(now), 2),
            "above_50d": bool(now > ma50), "above_200d": bool(now > ma200)}


# ----------------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------------
st.title("📄 Paper Trading Account")
st.caption("Read-only view of the paper trader. It never trades — the GitHub "
           "Action does that on a schedule.")

top = st.columns([1, 1, 1, 1, 1])
if top[4].button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

state = load_json(STATE_FILE)
trades = load_csv(LOG_FILE)

if state is None:
    st.warning("No account found yet. The trader has not run, or "
               f"`{STATE_FILE}` is not in the repo.")
    st.stop()

positions = state.get("positions", {})
prices = live_prices(tuple(sorted(positions.keys())))

# ---- headline ----
total_val = float(state.get("cash", 0.0))
for t, p in positions.items():
    total_val += p.get("shares", 0.0) * prices.get(t, p.get("avg_price", 0.0))
ret = (total_val / STARTING_CASH - 1) * 100 if STARTING_CASH else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total value", f"${total_val:,.2f}", f"{ret:+.2f}% since start")
c2.metric("Cash", f"${state.get('cash', 0):,.2f}")
c3.metric("Positions", len(positions))
c4.metric("Runs", state.get("runs", 0))

# ---- timing ----
t = timing_now()
spent = float(state.get("month_spent", 0.0))
mkey = state.get("month", "") or "n/a"

left, right = st.columns([2, 3])
with left:
    st.subheader("Market timing")
    if t:
        gate = "above" if t["score"] >= MIN_TIMING_SCORE else "below"
        colour = "green" if t["score"] >= MIN_TIMING_SCORE else "orange"
        st.markdown(f"### :{colour}[{t['score']}/100]")
        st.write(f"Gate is **{MIN_TIMING_SCORE}+** to invest — currently **{gate}** it.")
        st.caption(f"VIX {t['vix']} · S&P 500 {t['sp500']:,} · "
                   f"above 50d: {'yes' if t['above_50d'] else 'no'} · "
                   f"above 200d: {'yes' if t['above_200d'] else 'no'}")
        st.progress(min(1.0, t["score"] / 100))
    else:
        st.info("Could not read the market indicators right now.")
    st.caption("The score is a rule, not a probability.")

with right:
    st.subheader(f"Monthly budget — {mkey}")
    pct = (spent / MONTHLY_PURCHASE_LIMIT) if MONTHLY_PURCHASE_LIMIT else 0
    b1, b2 = st.columns(2)
    b1.metric("Spent this month", f"${spent:,.0f}")
    b2.metric("Remaining", f"${max(0, MONTHLY_PURCHASE_LIMIT - spent):,.0f}")
    st.progress(min(1.0, pct))
    st.caption(f"{pct:.0%} of the ${MONTHLY_PURCHASE_LIMIT:,.0f} monthly purchase limit.")

st.divider()

# ---- positions ----
st.subheader("Positions")
if not positions:
    st.info("No positions yet. The trader only buys when the timing score "
            f"reaches {MIN_TIMING_SCORE}.")
else:
    rows = []
    for tkr, p in sorted(positions.items()):
        px = prices.get(tkr, p.get("avg_price", 0.0))
        cost = p.get("shares", 0.0) * p.get("avg_price", 0.0)
        val = p.get("shares", 0.0) * px
        pl = val - cost
        rows.append({
            "Ticker": tkr,
            "Shares": round(p.get("shares", 0.0), 3),
            "Avg cost": round(p.get("avg_price", 0.0), 2),
            "Price": round(px, 2),
            "Cost basis": round(cost, 2),
            "Value": round(val, 2),
            "P/L ($)": round(pl, 2),
            "P/L (%)": round((pl / cost * 100) if cost else 0.0, 1),
        })
    df = pd.DataFrame(rows).sort_values("Value", ascending=False)

    def colour_pl(v):
        if isinstance(v, (int, float)):
            return "color: #1a7f37; font-weight: 600" if v >= 0 else "color: #cf222e; font-weight: 600"
        return ""

    st.dataframe(
        df.style.format({
            "Shares": "{:,.3f}", "Avg cost": "${:,.2f}", "Price": "${:,.2f}",
            "Cost basis": "${:,.2f}", "Value": "${:,.2f}",
            "P/L ($)": "${:+,.2f}", "P/L (%)": "{:+.1f}%",
        }).map(colour_pl, subset=["P/L ($)", "P/L (%)"]),
        use_container_width=True, hide_index=True,
    )

# ---- trade history ----
st.subheader("Trade history")
if trades is None or trades.empty:
    st.info("No trades logged yet.")
else:
    show = trades.iloc[::-1].copy()
    cols = [c for c in ["timestamp_utc", "ticker", "shares", "price", "amount",
                        "timing_score", "approach"] if c in show.columns]
    show = show[cols].rename(columns={
        "timestamp_utc": "When (UTC)", "ticker": "Ticker", "shares": "Shares",
        "price": "Price", "amount": "Amount", "timing_score": "Timing",
        "approach": "Approach",
    })
    fmt = {}
    if "Shares" in show.columns:
        fmt["Shares"] = "{:,.3f}"
    if "Price" in show.columns:
        fmt["Price"] = "${:,.2f}"
    if "Amount" in show.columns:
        fmt["Amount"] = "${:,.2f}"
    if "Timing" in show.columns:
        fmt["Timing"] = "{:.0f}"
    st.dataframe(show.style.format(fmt), use_container_width=True, hide_index=True)
    st.caption(f"{len(show)} trades logged.")

st.divider()
st.caption("Simulation only — no real orders were placed and no brokerage account "
           "is connected. Prices come from Yahoo's free feed and may be delayed. "
           "Decision-support tool, not financial advice.")
