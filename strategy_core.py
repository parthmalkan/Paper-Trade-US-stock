"""
strategy_core.py — the ONE place the scoring logic lives.

Both dashboard.py and paper_trader.py import from this file, so the model
cannot drift between them. Change it here and both follow.

WHAT'S IN IT
------------
  SP500_SET, DOW30          index membership (bundled, no network)
  UNIVERSE                  the bundled ~500-name list
  SETTINGS                  weights, caps, gates — all in one block
  market_timing()           VIX + S&P trend -> 0-100 rule score
  score_universe(prices)    the five-factor scoring engine
  pick_diversified(df)      Approach 1
  pick_momentum(df)         Approach 2
  pick_optimized(df,prices) Approach 3

NOT in it: anything Streamlit. This module must import cleanly outside a
Streamlit runtime, which is why the UI stays in the two app files.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS — the single source of truth for both apps
# ----------------------------------------------------------------------------
STARTING_CASH = 100000.0        # per approach, in the paper trader
MIN_PRICE = 5.0                 # skip penny stocks

# Scoring
RISK_ADJUSTED_MOMENTUM = True   # divide momentum by each name's volatility
QUALITY_WEIGHT = 0.15           # influence of the quality factor
QUALITY_USE_NETWORK = False     # True = old per-ticker .info lookups (slow)

# Universe
AUTO_UNIVERSE_SIZE = 500
TIER_SP500 = 250
TIER_DOW = 30
TIER_GAINERS = 60
TIER_SMALLCAP = 60
TIER_ACTIVE = 60
TIER_ROBINHOOD = 0
ROBINHOOD_100 = ""              # paste Robinhood's list here (no API exists)

# Portfolio construction
TOP_N_CORE = 8
TOP_N_MOM = 6
CAP_CORE = 0.20
CAP_MOM = 0.20
OPT_CANDIDATES = 20
OPT_MAX_WEIGHT = 0.15
OPT_METHOD = "Hierarchical Risk Parity"
RISK_FREE = 0.045

# Paper-trader behaviour
MIN_HOURS_BETWEEN_TRADES = 24
MONTHLY_PURCHASE_LIMIT = 15000.0
# Per-run share = MONTHLY_PURCHASE_LIMIT / RUNS_PER_MONTH_EXPECTED.
# At 20 that was $750, which almost never reached the $5,000 daily ceiling.
# At 5 it is $3,000, so a strong reading deploys real size. Keep this in step
# with paper_trader.py, which reads it from here unless it sets its own.
RUNS_PER_MONTH_EXPECTED = 5
SELL_SCORE_PERCENTILE = 0.33    # sell when a name falls into the bottom third
MIN_HOLD_MONTHS = 3             # no rank-based exit before this
APPROACHES = [
    ("Approach 1", "paper_account_a1.json", "paper_trades_a1.csv"),
    ("Approach 2", "paper_account_a2.json", "paper_trades_a2.csv"),
    ("Approach 3", "paper_account_a3.json", "paper_trades_a3.csv"),
]
BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500"


# ----------------------------------------------------------------------------
# INDEX MEMBERSHIP (bundled -- these cannot be fetched reliably)
# ----------------------------------------------------------------------------
SP500_SET = {
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","AVGO","TSLA",
    "LLY","JPM","V","UNH","XOM","MA","COST","HD","PG","JNJ",
    "WMT","ABBV","NFLX","CRM","BAC","ORCL","CVX","MRK","KO","AMD",
    "PEP","TMO","LIN","ADBE","CSCO","ACN","MCD","ABT","GE","WFC",
    "QCOM","PM","DHR","VZ","INTU","TXN","AMGN","CAT","IBM","NOW",
    "ISRG","RTX","SPGI","UBER","GS","PFE","BX","NEE","HON","T",
    "COP","BKNG","LOW","BLK","SYK","TJX","MS","SBUX","PLD","MDT",
    "BMY","GILD","ADP","VRTX","MMC","CB","SCHW","AMT","CI","ELV",
    "DE","BA","LMT","SO","INTC","MDLZ","FI","ADI","REGN","ETN",
    "KLAC","LRCX","MU","PANW","PLTR","BSX","CIEN","COHR","ANET","CDNS",
    "SNPS","APH","MSI","ROP","FTNT","NXPI","MCHP","HPQ","DELL","WDC",
    "STX","GLW","CDW","IT","SWKS","ZBRA","EPAM","JNPR","FFIV","AKAM",
    "NTAP","QRVO","GEN","TER","KEYS","TRMB","GRMN","TDY","LDOS","LHX",
    "HPE","ON","SMCI","CRWD","SNOW","DDOG","ZS","TEAM","WDAY","MSTR",
    "APP","MPWR","ENPH","FSLR","DIS","CMCSA","TMUS","EA","TTWO","WBD",
    "OMC","IPG","LYV","MTCH","PARA","NWSA","FOXA","NKE","DHI","AZO",
    "ORLY","MAR","HLT","GM","F","ROST","LEN","PHM","YUM","DRI",
    "ULTA","BBY","EBAY","EXPE","LVS","MGM","RCL","CCL","NCLH","APTV",
    "BWA","TPR","RL","HAS","MO","CL","TGT","KMB","GIS","K",
    "HSY","SYY","ADM","STZ","KHC","CHD","CLX","EL","KR","TSN",
    "MKC","BG","CAG","LW","HRL","SJM","CPB","C","AON","ICE",
    "CME","USB","PNC","TFC","COF","BK","AIG","MET","PRU","AFL",
    "ALL","TRV","AJG","MCO","MSCI","FIS","GPN","PYPL","SYF","DFS",
    "FITB","HBAN","RF","KEY","CFG","MTB","NTRS","STT","WTW","CINF",
    "WRB","BRO","CBOE","NDAQ","AMP","RJF","VTRS","EW","A","IQV",
    "BIIB","MRNA","IDXX","RMD","MTD","WAT","BAX","CAH","GEHC","DXCM",
    "PODD","ALGN","COO","STE","HOLX","ZBH","SOLV","LH","UNP","GD",
    "NOC","ITW","EMR","PH","CSX","NSC","FDX","WM","RSG","CTAS",
    "PAYX","FAST","ODFL","CARR","OTIS","TT","JCI","ROK","CMI","PCAR",
    "GWW","IR","DOV","AME","SWK","SNA","VRSK","EFX","AXON","GEV",
    "PWR","URI","WAB","EXPD","CHRW","UAL","DAL","LUV","AAL","MAS",
    "ALLE","J","SLB","EOG","MPC","PSX","VLO","OXY","WMB","KMI",
    "OKE","HES","BKR","HAL","DVN","FANG","EQT","TRGP","CTRA","APA",
    "MRO","DUK","CEG","AEP","SRE","D","PCG","EXC","XEL","ED",
    "WEC","ES","AEE","DTE","PPL","FE","ETR","AES","CNP","CMS",
    "NI","LNT","EVRG","PNW","ATO","NRG","VST","EQIX","WELL","SPG",
    "PSA","O","DLR","CCI","EXR","AVB","EQR","VTR","IRM","SBAC",
    "ARE","BXP","KIM","REG","HST","MAA","UDR","CPT","ESS","INVH",
    "DOC","VICI","WY","SHW","APD","FCX","ECL","NEM","NUE","DOW",
    "PPG","DD","VMC","MLM","IFF","LYB","STLD","ALB","CE","EMN",
    "PKG","IP","AMCR","BALL","AVY","CF","MOS","FMC","JBL","VSH",
    "ONTO","CAMT","FORM","ACLS","MKSI","NOVT","SLAB","CRUS","POWI","DIOD",
    "MTSI","SITM","ALGM","SYNA","RMBS","LSCC","WOLF","AMKR","SANM","PLXS",
    "FN","AAON","EXLS","GDDY","OKTA","TWLO","HUBS","DBX","BOX","ZM",
    "DOCU","NCNO","BILL","PCTY","PAYC","MANH","TYL","JKHY","GWRE","BSY",
    "PTC","AZPN","DT","ESTC","NET","BMRN","NBIX","ALNY","UHS","MOH",
    "CNC","SSNC","BR","TROW","BEN","IBKR","HEI","TDG","TXT","HWM",
    "ROL","BURL","FLUT","WYNN","ABNB","DECK","USFD","MNST","KDP","ET",
    "EPD","MPLX","AA","CCJ","FRT","SPOT","CACI","BAH","MRVL","ARM",
}

DOW30 = [
    ("AAPL", "Apple Inc.", "Technology"),
    ("AMGN", "Amgen Inc.", "Health Care"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("AXP", "American Express", "Financials"),
    ("BA", "Boeing Co.", "Industrials"),
    ("CAT", "Caterpillar Inc.", "Industrials"),
    ("CRM", "Salesforce Inc.", "Technology"),
    ("CSCO", "Cisco Systems", "Technology"),
    ("CVX", "Chevron Corp.", "Energy"),
    ("DIS", "Walt Disney Co.", "Communication Services"),
    ("GS", "Goldman Sachs Group", "Financials"),
    ("HD", "Home Depot Inc.", "Consumer Discretionary"),
    ("HON", "Honeywell International", "Industrials"),
    ("IBM", "IBM Corp.", "Technology"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("KO", "Coca-Cola Co.", "Consumer Staples"),
    ("MCD", "McDonald's Corp.", "Consumer Discretionary"),
    ("MMM", "3M Company", "Industrials"),
    ("MRK", "Merck & Co.", "Health Care"),
    ("MSFT", "Microsoft Corp.", "Technology"),
    ("NKE", "Nike Inc.", "Consumer Discretionary"),
    ("NVDA", "NVIDIA Corp.", "Technology"),
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("SHW", "Sherwin-Williams", "Materials"),
    ("TRV", "Travelers Companies", "Financials"),
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("V", "Visa Inc.", "Financials"),
    ("VZ", "Verizon Communications", "Communication Services"),
    ("WMT", "Walmart Inc.", "Consumer Staples"),
]


# ----------------------------------------------------------------------------
# MARKET TIMING — a RULE, not a forecast
# ----------------------------------------------------------------------------
def market_timing():
    """VIX level + S&P 500 trend, scored 0-100 against plain thresholds.

    This tells you current conditions relative to historical norms. It says
    nothing about what happens next. Returns None if the data is unreachable.
    """
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        spx = yf.Ticker("^GSPC").history(period="260d")["Close"]
        spx_now = float(spx.iloc[-1])
        ma50 = float(spx.rolling(50).mean().iloc[-1])
        ma200 = float(spx.rolling(200).mean().iloc[-1])
    except Exception:
        return None

    trend_up = spx_now > ma50 and spx_now > ma200
    score = 50
    score += 20 if trend_up else -20
    score += 15 if vix < 20 else (0 if vix < 30 else -15)
    score = int(max(0, min(100, score)))

    if score >= 65:
        regime = "Favorable"
    elif score >= 40:
        regime = "Neutral"
    else:
        regime = "Cautious"

    return {
        "score": score,
        "regime": regime,
        "vix": round(float(vix), 2),
        "sp500": round(spx_now, 2),
        "above_50d": bool(spx_now > ma50),
        "above_200d": bool(spx_now > ma200),
    }


# ----------------------------------------------------------------------------
# SCORING ENGINE — five factors, ranked inside the universe
# ----------------------------------------------------------------------------
def score_universe(prices, quality_scores):
    """Turn a price frame into a scored table.

    prices          index = dates, columns = tickers, values = close
    quality_scores  {ticker: 0-100} from price behaviour, no network

    Returns a DataFrame with one row per ticker and a Composite Score.
    """
    rows = []
    for t in prices.columns:
        try:
            px = prices[t].dropna()
            n = len(px)
            if n < 20:
                continue
            last = float(px.iloc[-1])
            if last < MIN_PRICE:
                continue

            r_6mo = float(last / px.iloc[-126] - 1) if n >= 126 else float(last / px.iloc[0] - 1)
            r_1mo = float(last / px.iloc[-21] - 1) if n >= 21 else 0.0
            ma50 = float(px.rolling(50).mean().iloc[-1]) if n >= 50 else float(px.mean())
            ma200 = float(px.rolling(200).mean().iloc[-1]) if n >= 200 else float(px.mean())
            vol = float(px.pct_change().std() * np.sqrt(252)) if n > 2 else 0.0

            rows.append({
                "Ticker": t,
                "Price": last,
                "Return 12mo": r_6mo,
                "Return 6mo": r_6mo,
                "Return 1mo": r_1mo,
                "Volatility": vol,
                "Trend": int(last > ma50) + int(last > ma200),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if RISK_ADJUSTED_MOMENTUM:
        v = df["Volatility"].clip(lower=0.05)
        mom_12 = df["Return 12mo"] / v
        mom_6 = df["Return 6mo"] / v
    else:
        mom_12 = df["Return 12mo"]
        mom_6 = df["Return 6mo"]

    df["Momentum Score"] = (mom_12.rank(pct=True) * 0.6 +
                            mom_6.rank(pct=True) * 0.4) * 100
    df["Trend Score"] = df["Trend"] * 50
    df["Low-Vol Score"] = (1 - df["Volatility"].rank(pct=True)) * 100
    df["Quality Score"] = df["Ticker"].map(lambda t: quality_scores.get(t, 50.0))

    w = max(0.0, min(0.5, QUALITY_WEIGHT))
    k = 1.0 - w
    df["Composite Score"] = (df["Momentum Score"] * (0.35 * k)
                             + df["Trend Score"] * (0.25 * k)
                             + df["Low-Vol Score"] * (0.20 * k)
                             + df["Quality Score"] * w
                             + 50 * (0.20 * k))
    return df


def quality_from_prices(prices):
    """Quality 0-100 from price behaviour alone. No network calls.

    trend     - position within the 6-month range
    drawdown  - how far below its own peak it sits
    stability - inverse of volatility
    """
    out = {}
    for t in prices.columns:
        try:
            px = prices[t].dropna()
            if len(px) < 40:
                out[t] = 50.0
                continue
            hi, lo, last = float(px.max()), float(px.min()), float(px.iloc[-1])
            span = (hi - lo) or 1.0
            pos = (last - lo) / span
            dd = (last / hi - 1.0) if hi > 0 else 0.0
            vol = float(px.pct_change().std() * np.sqrt(252)) or 0.01
            out[t] = round(max(0.0, min(100.0, pos * 100)) * 0.40
                           + max(0.0, min(100.0, 100 + dd * 200)) * 0.35
                           + max(0.0, min(100.0, 100 - vol * 150)) * 0.25, 2)
        except Exception:
            out[t] = 50.0
    return out


# ----------------------------------------------------------------------------
# THE THREE APPROACHES
# ----------------------------------------------------------------------------
def pick_diversified(df):
    """Approach 1: highest composite names, score-weighted, capped."""
    sub = df.copy()
    sub = sub.nlargest(TOP_N_CORE, "Composite Score")
    if sub.empty:
        return sub
    w = sub["Composite Score"] / sub["Composite Score"].sum()
    sub["Weight"] = np.minimum(w, CAP_CORE)
    sub["Weight"] = sub["Weight"] / sub["Weight"].sum()
    return sub.sort_values("Weight", ascending=False)


def pick_momentum(df):
    """Approach 2: highest raw 6-month return, equal-weighted."""
    sub = df.nlargest(TOP_N_MOM, "Return 6mo").copy()
    if sub.empty:
        return sub
    sub["Weight"] = 1.0 / len(sub)
    return sub.sort_values("Return 6mo", ascending=False)


def pick_optimized(df, prices, method=None):
    """Approach 3: max-Sharpe / HRP / Black-Litterman across a shortlist."""
    method = method or OPT_METHOD
    try:
        from pypfopt import (EfficientFrontier, risk_models, expected_returns,
                             HRPOpt, BlackLittermanModel)
    except ImportError:
        return None, "PyPortfolioOpt is not installed."

    shortlist = df.nlargest(OPT_CANDIDATES, "Composite Score")["Ticker"].tolist()
    sub = prices[[t for t in shortlist if t in prices.columns]].dropna()
    if sub.shape[1] < 2 or len(sub) < 60:
        return None, "Not enough clean price history to optimize."

    mu = expected_returns.mean_historical_return(sub)
    S = risk_models.sample_cov(sub)

    try:
        if method == "Hierarchical Risk Parity":
            opt = HRPOpt(returns=sub.pct_change().dropna())
            opt.optimize()
            clean = opt.clean_weights()
            perf = opt.portfolio_performance(risk_free_rate=RISK_FREE)
        elif method == "Black-Litterman":
            views = df.set_index("Ticker").reindex(sub.columns)["Return 6mo"]
            bl = BlackLittermanModel(S, pi="market", market_caps=None,
                                     risk_aversion=1.0,
                                     absolute_views=views.dropna().to_dict(),
                                     omega="idzorek")
            ef = EfficientFrontier(bl.bl_returns(), S,
                                   weight_bounds=(0, OPT_MAX_WEIGHT))
            ef.max_sharpe(risk_free_rate=RISK_FREE)
            clean = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
        else:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
            ef.max_sharpe(risk_free_rate=RISK_FREE)
            clean = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
    except Exception:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, OPT_MAX_WEIGHT))
            ef.max_sharpe(risk_free_rate=RISK_FREE)
            clean = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE)
        except Exception as e2:
            return None, f"Optimizer could not solve: {e2}"

    rows = []
    for t, w in clean.items():
        if w and w > 0.0005:
            meta = df[df["Ticker"] == t]
            if meta.empty:
                continue
            meta = meta.iloc[0]
            rows.append({
                "Ticker": t,
                "Company": meta.get("Company", t),
                "Sector": meta.get("Sector", "-"),
                "Price": meta["Price"],
                "Return 6mo": meta["Return 6mo"],
                "Volatility": meta["Volatility"],
                "Weight": w,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return None, "Optimizer returned no usable weights."
    out = out.sort_values("Weight", ascending=False)
    out["Weight"] = out["Weight"] / out["Weight"].sum()

    exp_ret, exp_vol, sharpe = perf
    return out, {"return": exp_ret, "vol": exp_vol, "sharpe": sharpe,
                 "method": method}


def picks_for(approach, df, prices):
    """Dispatch to the right approach by name. Returns a DataFrame or None."""
    try:
        if approach.startswith("Approach 1"):
            return pick_diversified(df)
        if approach.startswith("Approach 2"):
            return pick_momentum(df)
        out, _info = pick_optimized(df, prices)
        return out
    except Exception:
        return None
