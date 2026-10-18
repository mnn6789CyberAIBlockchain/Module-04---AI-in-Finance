
"""Fetch market data for the FinSight AI dashboard."""
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests
import yfinance as yf


# Eight price-based indicators from Yahoo Finance
PRICE_MARKETS = [
    {"key": "sp500", "label": "S&P 500", "ticker": "^GSPC", "format": "index"},
    {"key": "qqq", "label": "NASDAQ-100 ETF (QQQ)", "ticker": "QQQ", "format": "money"},
    {"key": "dow", "label": "Dow Jones", "ticker": "^DJI", "format": "index"},
    {"key": "russell", "label": "Russell 2000", "ticker": "^RUT", "format": "index"},
    {"key": "vix", "label": "VIX", "ticker": "^VIX", "format": "index"},
    {"key": "gold", "label": "Gold Futures", "ticker": "GC=F", "format": "money"},
    {"key": "wti", "label": "WTI Crude Oil", "ticker": "CL=F", "format": "money"},
    {"key": "bitcoin", "label": "Bitcoin", "ticker": "BTC-USD", "format": "money"},
]


def unavailable_card(key, label, fmt, source="Quote unavailable"):
    """Return a placeholder when a data source cannot provide a quote."""
    return {
        "key": key,
        "label": label,
        "value": None,
        "change_pct": None,
        "format": fmt,
        "history": [],
        "source": source,
        "updated": None,
    }


def get_yahoo_card(item):
    """Fetch a latest available daily close and a short trend history."""
    ticker = yf.Ticker(item["ticker"])
    history = ticker.history(period="1mo", interval="1d")

    if history is None or history.empty:
        raise ValueError("No price history returned")

    closes = history["Close"].dropna()
    if closes.empty:
        raise ValueError("No closing prices returned")

    latest = float(closes.iloc[-1])
    previous = float(closes.iloc[-2]) if len(closes) >= 2 else latest

    change = (
        (latest - previous) / previous * 100
        if previous != 0 else 0.0
    )

    chart = [
        {
            "date": date.strftime("%b %d"),
            "value": round(float(value), 4),
        }
        for date, value in closes.items()
    ]

    return {
        "key": item["key"],
        "label": item["label"],
        "value": latest,
        "change_pct": change,
        "format": item["format"],
        "history": chart,
        "source": "Yahoo Finance via yfinance",
        "updated": closes.index[-1].strftime("%Y-%m-%d"),
    }


def get_fred_card(series_id, label):
    """Fetch daily US Treasury yields from the FRED public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if df.empty or len(df.columns) < 2:
        raise ValueError("No FRED data returned")

    date_col, value_col = df.columns[:2]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if df.empty:
        raise ValueError("No valid Treasury yield observations")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest

    chart = [
        {"date": str(row[date_col])[-5:],
         "value": round(float(row[value_col]), 4)}
        for _, row in df.tail(30).iterrows()
    ]

    return {
        "key": series_id.lower(),
        "label": label,
        "value": float(latest[value_col]),
        # Yield change is measured in percentage points, not percent.
        "change_pct": float(latest[value_col]) - float(previous[value_col]),
        "format": "yield",
        "history": chart,
        "source": "FRED / Federal Reserve Bank of St. Louis",
        "updated": str(latest[date_col]),
    }


def get_market_dashboard():
    """Return all ten cards, even if one or more sources fail."""
    cards = []

    for item in PRICE_MARKETS:
        try:
            cards.append(get_yahoo_card(item))
        except Exception as exc:
            print(f"Market data warning ({item['ticker']}): {exc}")
            cards.append(
                unavailable_card(item["key"], item["label"], item["format"])
            )

    treasury_series = [
        ("DGS10", "US 10-Year Treasury Yield"),
        ("DGS2", "US 2-Year Treasury Yield"),
    ]

    for series_id, label in treasury_series:
        try:
            cards.append(get_fred_card(series_id, label))
        except Exception as exc:
            print(f"Treasury data warning ({series_id}): {exc}")
            cards.append(
                unavailable_card(series_id.lower(), label, "yield", "FRED unavailable")
            )

    # Keep cards in the requested dashboard order.
    order = [
        "sp500", "qqq", "dow", "russell", "vix",
        "dgs10", "dgs2", "gold", "wti", "bitcoin",
    ]
    by_key = {card["key"]: card for card in cards}

    return {
        "cards": [by_key[key] for key in order],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
