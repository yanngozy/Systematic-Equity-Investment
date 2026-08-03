import json
import pyarrow.parquet as pq
import pyarrow as pa
import time
import gc
import requests
import ijson
import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal

"""
    Build a low-sparsity, point-in-time fundamentals panel from SEC EDGAR XBRL data
    and merge it onto a daily price panel with no look-ahead.
"""


# ------------------------------------------------------------------------
# CONFIG -- fill this in
# ------------------------------------------------------------------------
USER_AGENT = "email@gmail.com"                     # <-- REQUIRED: put a real contact here
CACHE_DIR = Path("./edgar_cache")                   # raw JSON responses cached locally
CACHE_DIR.mkdir(exist_ok=True)
REQUEST_DELAY = 0.11                                # stay under SEC's 10 req/sec limit

HEADERS = {"User-Agent": USER_AGENT}

# ------------------------------------------------------------------------
# Concept map: priority-ordered fallback tags per financial concept.
# XBRL tag usage drifts over time and across companies (e.g. after ASC 606,
# many switched Revenues -> RevenueFromContractWithCustomerExcludingAssessedTax).
# Pulling only ONE tag per concept is the #1 cause of sparsity -- this unions
# several known-equivalent tags, preferring the first available in priority order.
# ------------------------------------------------------------------------
CONCEPT_MAP = {
    "TotalAssets":      {"tags": ["Assets"], "kind": "instant"},
    "TotalLiabilities": {"tags": ["Liabilities"], "kind": "instant"},
    "Equity":           {"tags": ["StockholdersEquity",
                                  "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                          "kind": "instant"},
    "Cash":             {"tags": ["CashAndCashEquivalentsAtCarryingValue",
                                  "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                          "kind": "instant"},
    "LongTermDebt":     {"tags": ["LongTermDebtNoncurrent", "LongTermDebt"], "kind": "instant"},
    
    "Revenue":          {"tags": ["Revenues",
                                  "RevenueFromContractWithCustomerExcludingAssessedTax",
                                  "RevenueFromContractWithCustomerIncludingAssessedTax",
                                  "SalesRevenueNet"],
                          "kind": "duration"},
    
    "DividendsPaid":    {   "tags": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
                            "kind": "duration",
                            "unit": "USD"
                        },
    
    "NetIncome":        {"tags": ["NetIncomeLoss", "ProfitLoss"], "kind": "duration"},
    "OperatingIncome":  {"tags": ["OperatingIncomeLoss"], "kind": "duration"},
    "GrossProfit":      {"tags": ["GrossProfit"], "kind": "duration"},
    "OperatingCF":      {"tags": ["NetCashProvidedByUsedInOperatingActivities"], "kind": "duration"},
    "CapEx":            {"tags": ["PaymentsToAcquirePropertyPlantAndEquipment"], "kind": "duration"},
    "RnD":              {"tags": ["ResearchAndDevelopmentExpense"], "kind": "duration"},
    "EPS_Basic":        {"tags": ["EarningsPerShareBasic"], "kind": "duration", "unit": "USD/shares"},
    "EPS_Diluted":      {"tags": ["EarningsPerShareDiluted"], "kind": "duration", "unit": "USD/shares"},
    "SharesOutstanding": {"tags": ["EntityCommonStockSharesOutstanding"], "kind": "instant",
                          "unit": "shares", "namespace": "dei",
                          "fallback_tags": ["CommonStockSharesOutstanding"]},
}

# ------------------------------------------------------------------------
# CIK lookup
# ------------------------------------------------------------------------
def get_ticker_to_cik() -> dict:
    """SEC's official ticker->CIK mapping. Small file, fetch once and cache."""
    cache_path = CACHE_DIR / "company_tickers.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
    else:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
        r.raise_for_status()
        raw = r.json()
        cache_path.write_text(json.dumps(raw))
        
    # Build base dictionary from the SEC JSON file
    mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
    
    # ------------------------------------------------------------------------
    # FIXED: HARDCODED OVERRIDES FOR MISSING S&P 500 TICKERS / SEC DISCREPANCIES
    # ------------------------------------------------------------------------
    overrides = {
        "BK": "0001390777",    # The Bank of New York Mellon Corp
        "CTRA": "0000858470",  # Coterra Energy Inc.
        "HOLX": "0000859747",  # Hologic, Inc.
        
        # Structural Class Share Suffixes (Often missed or mismatched by the SEC map)
        "BRK.B": "0001067983", # Berkshire Hathaway Class B
        "BRK.A": "0001067983", # Berkshire Hathaway Class A
        "BF.B": "0000014693",  # Brown-Forman Corp Class B
        "BF.A": "0000014693",  # Brown-Forman Corp Class A
        "LEN.B": "0000920743", # Lennar Corp Class B
        "JWN": "0000072333"    # Nordstrom (In case of mapping lags)
    }
    
    mapping.update({k.upper(): v for k, v in overrides.items()})
   
    return mapping

def fetch_company_facts(cik: str) -> dict:
    """Fetch + cache the full XBRL companyfacts payload for one CIK."""
    cache_path = CACHE_DIR / f"{cik}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    time.sleep(REQUEST_DELAY)
    if r.status_code != 200:
        return {}
    data = r.json()
    cache_path.write_text(json.dumps(data))
    return data


# ------------------------------------------------------------------------
# Parsing / sparsity reduction
# ------------------------------------------------------------------------
def _facts_for_tag(company_facts: dict, tag: str, unit: str = "USD", namespace: str = "us-gaap") -> pd.DataFrame:
    try:
        units = company_facts["facts"][namespace][tag]["units"]
    except KeyError:
        return pd.DataFrame()
    if unit not in units:
        return pd.DataFrame()
    df = pd.DataFrame(units[unit])
    if df.empty:
        return df
    df["tag"] = tag
    return df


def _quarterly_only(df: pd.DataFrame) -> pd.DataFrame:
    """Duration tags mix quarterly + YTD/annual entries. Keep only ~90-day windows
    so e.g. a Q3 filing's 9-month YTD figure doesn't get mistaken for the quarter."""
    if df.empty or "start" not in df.columns:
        return df
    d = df.copy()
    d["start"] = pd.to_datetime(d["start"])
    d["end"] = pd.to_datetime(d["end"])
    d["span_days"] = (d["end"] - d["start"]).dt.days
    return d[(d["span_days"] >= 80) & (d["span_days"] <= 100)].drop(columns="span_days")


def extract_concept(company_facts: dict, concept: str) -> pd.DataFrame:
    spec = CONCEPT_MAP[concept]
    unit = spec.get("unit", "USD")
    namespace = spec.get("namespace", "us-gaap")
    frames = []
    for tag in spec["tags"]:
        f = _facts_for_tag(company_facts, tag, unit=unit, namespace=namespace)
        if spec["kind"] == "duration":
            f = _quarterly_only(f)
        if not f.empty:
            frames.append(f)
    # fallback_tags may live in a different namespace (e.g. dei tag missing -> try us-gaap)
    for tag in spec.get("fallback_tags", []):
        f = _facts_for_tag(company_facts, tag, unit=unit, namespace="us-gaap")
        if spec["kind"] == "duration":
            f = _quarterly_only(f)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["end", "filed", concept])

    combined = pd.concat(frames, ignore_index=True)
    combined["end"] = pd.to_datetime(combined["end"])
    combined["filed"] = pd.to_datetime(combined["filed"])

    # Restatement handling: keep the value as FIRST FILED for each fiscal end-date
    # tags when two tags both cover the same end-date.
    tag_priority = {t: i for i, t in enumerate(spec["tags"] + spec.get("fallback_tags", []))}
    combined["tag_rank"] = combined["tag"].map(tag_priority)
    dedup = (combined.sort_values(["end", "tag_rank", "filed"])
                      .drop_duplicates(subset=["end"], keep="first"))

    return dedup[["end", "filed", "val"]].rename(columns={"val": concept})


def build_ticker_fundamentals(company_facts: dict) -> pd.DataFrame:
    """One row per (end, filed) fiscal period, all concepts unioned on 'end'."""
    panel = None
    for concept in CONCEPT_MAP:
        c = extract_concept(company_facts, concept)
        if c.empty:
            continue
        # Merge key is 'filed' only -- that's the actual point-in-time timeline.
        # Two concepts can be filed on the same date but reference different fiscal
        # 'end' dates (e.g. a 10-Q's balance-sheet date vs. the cover page's
        # shares-outstanding-as-of date); keying on ('end','filed') together would
        # split those into separate sparse rows instead of one combined snapshot.
        c = c.drop(columns=["end"]).rename(columns={concept: concept})
        panel = c if panel is None else panel.merge(c, on="filed", how="outer")
        
    if panel is None:
        return pd.DataFrame()
        
    panel = panel.sort_values("filed").reset_index(drop=True)
    # Forward-fill within the fundamentals timeline itself: each filing event should
    # carry forward every previously-known concept, not just the one(s) that happen
    # to have been re-reported in that exact filing.
    concept_cols = [c for c in panel.columns if c != "filed"]
    panel[concept_cols] = panel[concept_cols].ffill()
    # Collapse multiple rows sharing the same 'filed' timestamp into one (keep the
    # last, now fully forward-filled, row for each).
    panel = panel.drop_duplicates(subset="filed", keep="last").reset_index(drop=True)
    return panel

def merge_point_in_time(prices: pd.DataFrame, fundamentals: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward-fill fundamentals onto the daily price panel using ONLY data already
    filed as of each date (merge_asof backward). This is what removes sparsity
    without introducing look-ahead: every row gets the latest already-public
    snapshot instead of only the single exact report-date row."""
    p = prices.sort_values("Date").copy()
    if fundamentals.empty:
        for concept in CONCEPT_MAP:
            p[concept] = np.nan
        p["MarketCap"] = np.nan
        p["Ticker"] = ticker
        return p
        
    f = fundamentals.sort_values("filed").copy()

    cols = [    'TotalAssets','TotalLiabilities', 'Equity', 'RnD', 
                'LongTermDebt', 'Revenue','NetIncome', 
                'OperatingIncome', 'OperatingCF'
           ]
    
    cols += 'EPS_Diluted' if 'EPS_Diluted' in f.columns else 'EPS_Basic'
    
    for col in f.columns:
        col_name = col
        if col not in cols:
            continue
        elif col=="EPS_Diluted" or col=="EPS_Basic":
            col_name = "EPS"
        elif col=="TotalAssets":
            col_name = "Assets"
        for w in [1,2,4]:
            f[col_name + "_Growth_" f"{3*w}m"] = f[col].pct_change(w)
   
    merged = pd.merge_asof(p, f, left_on="Date", right_on="filed", direction="backward")
    # MarketCap needs a DAILY price, so it can't be computed inside the fundamentals
    # every trading day even though shares outstanding only updates once a quarter.
    if "SharesOutstanding" in merged.columns:
        merged["MarketCap"] = merged["SharesOutstanding"] * merged["Close"]
    else:
        merged["MarketCap"] = np.nan
    merged["Ticker"] = ticker
    return merged


# ------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------
def build_fundamentals_for_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Returns {ticker: fundamentals_panel_df}. Cached on disk, safe to re-run."""
    ticker_to_cik = get_ticker_to_cik()
    out = {}
    for i, t in enumerate(tickers, 1):
        cik = ticker_to_cik.get(t.upper())
        if cik is None:
            print(f"[{i}/{len(tickers)}] {t}: no CIK found, skipping")
            out[t] = pd.DataFrame()
            continue
        facts = fetch_company_facts(cik)
        panel = build_ticker_fundamentals(facts)
        if i%50==0:
            print(f"[{i}/{len(tickers)}] {t}: {len(panel)} fiscal periods")
        out[t] = panel
    return out


def attach_to_price_panel(prices: pd.DataFrame, fundamentals_by_ticker: dict) -> pd.DataFrame:
    """prices must have columns ['Date','Ticker', ...]. Returns prices + fundamental columns."""
    pieces = []
    for ticker, group in prices.groupby("Ticker"):
        fund = fundamentals_by_ticker.get(ticker, pd.DataFrame())
        pieces.append(merge_point_in_time(group.drop(columns=["Ticker"]), fund, ticker))
    return pd.concat(pieces, ignore_index=True)

def add_fundamentals_to_pv_dataset(df: pd.DataFrame):
    
    raw = df.copy()

    price_volume_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
                          "returns", "shortName", "sector", "industry"]
  
    prices = raw[price_volume_cols].copy()
    tickers = sorted(prices["Ticker"].unique())
  
    for x in [tickers[267],tickers[407]]:
        tickers.remove(x)

    fundamentals_by_ticker = build_fundamentals_for_tickers(tickers)
    full_panel = attach_to_price_panel(prices, fundamentals_by_ticker)

    fundamental_cols = list(CONCEPT_MAP.keys()) + ["MarketCap"]

    return full_panel
