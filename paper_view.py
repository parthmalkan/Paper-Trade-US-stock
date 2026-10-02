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

DEPLOY
------
1. Put this file, paper_trader.py, requirements.txt and the account files in
   one GitHub repo.
2. share.streamlit.io -> New app -> pick the repo -> main file: paper_view.py
3. Set REPO_RAW_BASE below so the page reads the newest committed data.

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
# SETTINGS — keep in step with paper_trader.py
# ----------------------------------------------------------------------------
STARTING_CASH = 100000.0
MONTHLY_PURCHASE_LIMIT = 15000.0
MIN_TIMING_SCORE = 80
MIN_HOURS_BETWEEN_TRADES = 24
BETTER_SCORE_WINDOW_HOURS = 72

APPROACHES = [
    ("Approach 1", "Diversified large-cap",
     "paper_account_a1.json", "paper_trades_a1.csv"),
    ("Approach 2", "Momentum / speculative",
     "paper_account_a2.json", "paper_trades_a2.csv"),
    ("Approach 3", "Markowitz optimizer",
     "paper_account_a3.json", "paper_trades_a3.csv"),
]

# Set this to your repo's raw URL so the page always reads the newest commit.
# Leave blank to read the local copies that came with the deploy.
REPO_RAW_BASE = "[raw.githubusercontent.com](https://raw.githubusercontent.com/parthmalkan/Paper-Trade-US-stock/main)"

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
    gate = "above" if timing["score"] >= MIN_TIMING_SCORE else "below"
    colour = "green" if timing["score"] >= MIN_TIMING_SCORE else "orange"
    st.markdown(
        f"**Market timing: :{colour}[{timing['score']}/100]** — {gate} the "
        f"{MIN_TIMING_SCORE} gate. "
        f"VIX {timing['vix']} · S&P 500 {timing['sp500']:,} · "
        f"above 50d: {'yes' if timing['above_50d'] else 'no'} · "
        f"above 200d: {'yes' if timing['above_200d'] else 'no'}"
    )
    st.progress(min(1.0, timing["score"] / 100))
    st.caption("The score is a rule, not a probability. Below the gate, no "
               "approach trades at all.")
else:
    st.info("Could not read the market indicators right now.")

st.divider()

# Load every account once
loaded = []
for name, blurb, state_file, log_file in APPROACHES:
    stt = load_json(state_file)
    trd = load_csv(log_file)
    loaded.append({"name": name, "blurb": blurb, "state": stt, "trades": trd})

all_tickers = sorted({t for a in loaded
                      for t in (a["state"].get("positions", {}) if a["state"] else {})})
prices = live_prices(tuple(all_tickers))

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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total value", f"${total:,.2f}", f"{ret:+.2f}%")
        c2.metric("Cash", f"${cash:,.2f}")
        c3.metric("Positions", len(positions))
        c4.metric("Runs", a["state"].get("runs", 0))

        spent = float(a["state"].get("month_spent", 0.0))
        mkey = a["state"].get("month", "") or "n/a"
        st.markdown("**Budget — " + mkey + "**")
        b1, b2 = st.columns(2)
        b1.metric("Spent this month", f"${spent:,.0f}")
        b2.metric("Remaining", f"${max(0, MONTHLY_PURCHASE_LIMIT - spent):,.0f}")
        _pct = (spent / MONTHLY_PURCHASE_LIMIT) if MONTHLY_PURCHASE_LIMIT else 0
        st.progress(min(1.0, _pct))
        st.caption(f"{_pct:.0%} of ${MONTHLY_PURCHASE_LIMIT:,.0f} used.")

        last = a["state"].get("last_trade_utc", "") or "never"
        last_score = a["state"].get("last_trade_score", 0)
        _line = "Last trade: **" + last + "**"
        if last != "never":
            _line += f" at score {last_score}"
        _line += (f" · Rules: max 1 trade / {MIN_HOURS_BETWEEN_TRADES}h, and "
                  f"within {BETTER_SCORE_WINDOW_HOURS}h only if the score improves.")
        st.caption(_line)

        st.markdown("**Positions**")
        if not rows:
            st.info("No positions yet — this approach has not traded.")
        else:
            dfp = pd.DataFrame(rows).sort_values("Value", ascending=False)
            st.dataframe(
                dfp.style.format({
                    "Shares": "{:,.3f}", "Avg cost": "${:,.2f}", "Price": "${:,.2f}",
                    "Cost basis": "${:,.2f}", "Value": "${:,.2f}",
                    "P/L ($)": "${:+,.2f}", "P/L (%)": "{:+.1f}%",
                }).map(colour_pl, subset=["P/L ($)", "P/L (%)"]),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Trade history**")
        if a["trades"] is None or a["trades"].empty:
            st.info("No trades logged yet.")
        else:
            sh = a["trades"].iloc[::-1]
            cols = [c for c in ["timestamp_utc", "ticker", "shares", "price",
                                "amount", "timing_score"] if c in sh.columns]
            sh = sh[cols].rename(columns={
                "timestamp_utc": "When (UTC)", "ticker": "Ticker",
                "shares": "Shares", "price": "Price", "amount": "Amount",
                "timing_score": "Timing"})
            fmt = {"Shares": "{:,.3f}", "Price": "${:,.2f}",
                   "Amount": "${:,.2f}", "Timing": "{:.0f}"}
            st.dataframe(sh.style.format(fmt), use_container_width=True, hide_index=True)
            st.caption(f"{len(sh)} trades.")

# ---------------- summary tab ----------------
with tabs[len(loaded)]:
    st.subheader("🏆 Which approach is best so far?")
    st.caption("Ranked by total return, which blends what each approach bought "
               "with the cash it left idle.")

    comp = []
    for a in loaded:
        if a["state"] is None:
            comp.append({"Approach": a["name"], "Total value": None,
                         "Return (%)": None, "Cash": None, "Positions": 0,
                         "Trades": 0, "Last trade": "-"})
            continue
        total, ret, cash, rows, positions = account_snapshot(a["state"], prices)
        ntrades = 0 if a["trades"] is None or a["trades"].empty else len(a["trades"])
        comp.append({
            "Approach": a["name"],
            "Total value": round(total, 2),
            "Return (%)": round(ret, 2),
            "Cash": round(cash, 2),
            "Positions": len(positions),
            "Trades": ntrades,
            "Last trade": a["state"].get("last_trade_utc", "") or "never",
        })

    cdf = pd.DataFrame(comp)
    ranked = cdf.dropna(subset=["Return (%)"]).sort_values("Return (%)", ascending=False)

    if ranked.empty:
        st.info("No account has traded yet, so there is nothing to rank. All "
                f"{len(loaded)} approaches are waiting for the score to clear "
                f"{MIN_TIMING_SCORE}.")
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

        st.markdown("**Why this ranking**")
        st.markdown(
            f"- **{best['Approach']}** leads on total return, which counts both "
            "the positions it holds and the cash it kept uninvested. An approach "
            "that trades often but holds cash earns less on that idle money.\n"
            "- A high cash balance usually means the 80 timing gate kept it out "
            "of the market — disciplined in a falling market, costly in a rising "
            "one.\n"
            "- Trade count matters twice: every trade is a decision, and the "
            "fewer trades needed for the same return, the less the approach "
            "relied on activity to get there.\n"
            "- A short history is not evidence. Give all three approaches the "
            "same stretch of market before drawing conclusions."
        )

        st.caption("Rankings come from the committed account files. If an "
                   "approach shows no trades, that is the 80 gate keeping it "
                   "out, not an error.")

st.divider()
st.caption("Simulation only — no real orders were placed and no brokerage "
           "account is connected. Prices come from Yahoo's free feed and may be "
           "delayed. Decision-support tool, not financial advice.")
