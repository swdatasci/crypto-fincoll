#!/usr/bin/env python3
"""
Alpha Vantage API Client - Direct REST API wrapper for feature extraction

Provides programmatic access to Alpha Vantage data for V5 training:
- News sentiment
- Company fundamentals
- Technical indicators (pre-computed)
- Economic indicators
- Cross-asset data (forex, commodities, crypto)

Note: This uses direct REST API calls, NOT the MCP server.
The MCP server (uvx av-mcp) is for interactive LLM/agent use in Claude Code.
This client is optimized for batch processing during training data generation.
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Client for Alpha Vantage REST API (direct HTTP calls)"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage API client.

        API key resolution order:
        1. Explicit ``api_key`` argument
        2. ``ALPHA_VANTAGE_API_KEY`` environment variable
        3. Credentials JSON file at ``$CREDENTIALS_DIR/.alpha_vantage_credentials.json``
           (CREDENTIALS_DIR defaults to ~/caelum/ss)

        Args:
            api_key: Alpha Vantage API key
        """
        import json
        import os
        from pathlib import Path

        if api_key is None:
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if api_key is None:
            creds_dir = Path(
                os.getenv("CREDENTIALS_DIR", str(Path.home() / "caelum" / "ss"))
            )
            creds_file = creds_dir / ".alpha_vantage_credentials.json"
            if creds_file.exists():
                try:
                    data = json.loads(creds_file.read_text())
                    api_key = data.get("api_key") or data.get("key")
                except Exception as e:
                    logger.warning(
                        f"Could not read Alpha Vantage credentials file: {e}"
                    )
            else:
                logger.warning(
                    f"Alpha Vantage credentials not found at {creds_file}. "
                    "Set ALPHA_VANTAGE_API_KEY env var or CREDENTIALS_DIR."
                )

        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_news_sentiment(
        self,
        tickers: str,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 50,
        sort: str = "LATEST",
    ) -> Dict[str, Any]:
        """
        Get news sentiment for ticker(s)

        Args:
            tickers: Comma-separated ticker symbols (e.g., "AAPL,MSFT")
            time_from: Start time for news (default: 24 hours ago)
            time_to: End time for news (default: now)
            limit: Max number of articles (default: 50)
            sort: Sort order - LATEST, EARLIEST, RELEVANCE (default: LATEST)

        Returns:
            {
                'feed': [
                    {
                        'title': str,
                        'url': str,
                        'time_published': str,
                        'summary': str,
                        'source': str,
                        'overall_sentiment_score': float (-1 to +1),
                        'overall_sentiment_label': str,
                        'ticker_sentiment': [
                            {
                                'ticker': str,
                                'relevance_score': float (0 to 1),
                                'ticker_sentiment_score': float (-1 to +1),
                                'ticker_sentiment_label': str
                            }
                        ]
                    }
                ]
            }
        """
        import requests

        # Format timestamps if provided
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "limit": limit,
            "sort": sort,
            "apikey": self.api_key,
        }

        if time_from:
            params["time_from"] = time_from.strftime("%Y%m%dT%H%M")
        if time_to:
            params["time_to"] = time_to.strftime("%Y%m%dT%H%M")

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")

        return data

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company fundamentals overview

        Args:
            symbol: Stock ticker symbol

        Returns:
            {
                'Symbol': str,
                'Name': str,
                'MarketCapitalization': str,
                'EBITDA': str,
                'PERatio': str,
                'PEGRatio': str,
                'BookValue': str,
                'DividendPerShare': str,
                'DividendYield': str,
                'EPS': str,
                'RevenuePerShareTTM': str,
                'ProfitMargin': str,
                'OperatingMarginTTM': str,
                'ReturnOnAssetsTTM': str,
                'ReturnOnEquityTTM': str,
                'RevenueTTM': str,
                'GrossProfitTTM': str,
                'QuarterlyEarningsGrowthYOY': str,
                'QuarterlyRevenueGrowthYOY': str,
                'AnalystTargetPrice': str,
                'TrailingPE': str,
                'ForwardPE': str,
                'PriceToSalesRatioTTM': str,
                'PriceToBookRatio': str,
                'EVToRevenue': str,
                'EVToEBITDA': str,
                'Beta': str,
                '52WeekHigh': str,
                '52WeekLow': str,
                '50DayMovingAverage': str,
                '200DayMovingAverage': str
            }
        """
        import requests

        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key}

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")

        return data

    def get_intraday_bars(
        self, symbol: str, interval: str = "1min", outputsize: str = "compact"
    ) -> Dict[str, Any]:
        """
        Get intraday OHLCV bars

        Args:
            symbol: Stock ticker
            interval: 1min, 5min, 15min, 30min, 60min
            outputsize: compact (100 bars) or full (extended history)

        Returns:
            {
                'Meta Data': {...},
                'Time Series (1min)': {
                    '2025-11-02 16:00:00': {
                        '1. open': str,
                        '2. high': str,
                        '3. low': str,
                        '4. close': str,
                        '5. volume': str
                    }
                }
            }
        """
        import requests

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")

        return data

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "5min",
        time_period: int = 14,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get pre-computed technical indicator

        Args:
            symbol: Stock ticker
            indicator: Indicator name (RSI, MACD, BBANDS, etc.)
            interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            time_period: Number of data points for calculation
            **kwargs: Additional indicator-specific parameters

        Returns:
            {
                'Meta Data': {...},
                'Technical Analysis: {INDICATOR}': {
                    '2025-11-02 16:00:00': {
                        '{INDICATOR}': str
                    }
                }
            }
        """
        import requests

        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "apikey": self.api_key,
        }
        params.update(kwargs)

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")

        return data

    def get_real_gdp(self, interval: str = "quarterly") -> Dict[str, Any]:
        """Get Real GDP economic indicator"""
        import requests

        params = {"function": "REAL_GDP", "interval": interval, "apikey": self.api_key}

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def get_treasury_yield(
        self, interval: str = "daily", maturity: str = "10year"
    ) -> Dict[str, Any]:
        """
        Get Treasury Yield

        Args:
            interval: daily, weekly, monthly
            maturity: 3month, 2year, 5year, 7year, 10year, 30year
        """
        import requests

        params = {
            "function": "TREASURY_YIELD",
            "interval": interval,
            "maturity": maturity,
            "apikey": self.api_key,
        }

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Get real-time forex exchange rate"""
        import requests

        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.api_key,
        }

        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()


# Feature extraction helpers


def extract_news_sentiment_features(
    sentiment_data: Dict[str, Any], symbol: str
) -> Dict[str, float]:
    """
    Extract 20D news sentiment features from Alpha Vantage response

    Args:
        sentiment_data: Response from get_news_sentiment()
        symbol: Target ticker symbol

    Returns:
        Dictionary with f81-f100 (20D news features)
    """
    features = {}

    articles = sentiment_data.get("feed", [])

    if not articles:
        # Return zeros if no news
        return {f"f{i}": 0.0 for i in range(81, 101)}

    # f81: Overall sentiment score (weighted average)
    sentiments = []
    relevances = []

    for article in articles:
        ticker_sentiments = article.get("ticker_sentiment", [])
        for ts in ticker_sentiments:
            if ts.get("ticker") == symbol:
                sentiments.append(float(ts.get("ticker_sentiment_score", 0)))
                relevances.append(float(ts.get("relevance_score", 0)))

    if sentiments:
        # Weighted average by relevance
        total_relevance = sum(relevances)
        features["f81"] = (
            sum(s * r for s, r in zip(sentiments, relevances)) / total_relevance
            if total_relevance > 0
            else 0
        )
    else:
        features["f81"] = 0.0

    # f82: News volume (article count)
    features["f82"] = len(articles) / 50.0  # Normalize to 0-1 (assuming max 50)

    # f83: Source credibility (average)
    # TODO: Implement source credibility scoring
    features["f83"] = 0.5  # Placeholder

    # f84: Topic finance percentage
    # TODO: Parse topics from Alpha Vantage response
    features["f84"] = 0.5  # Placeholder

    # f85: Topic company-specific percentage
    features["f85"] = 0.5  # Placeholder

    # f86: News surprise factor (spike in volume)
    # TODO: Compare to historical baseline
    features["f86"] = 0.0  # Placeholder

    # f87: Sentiment change (acceleration)
    # TODO: Compare to previous period
    features["f87"] = 0.0  # Placeholder

    # f88: Controversy score (variance in sentiment)
    if len(sentiments) > 1:
        import numpy as np

        features["f88"] = float(np.std(sentiments))
    else:
        features["f88"] = 0.0

    # f89: News urgency (recency weighting)
    recent_count = sum(
        1
        for a in articles
        if datetime.fromisoformat(
            a["time_published"].replace("T", " ").replace("Z", "")
        )
        > datetime.now() - timedelta(hours=4)
    )
    features["f89"] = recent_count / len(articles) if articles else 0.0

    # f90: Consensus (percentage positive)
    if sentiments:
        features["f90"] = sum(1 for s in sentiments if s > 0.15) / len(sentiments)
    else:
        features["f90"] = 0.5

    # f91-f100: Reserved for future expansion or other features
    for i in range(91, 101):
        features[f"f{i}"] = 0.0

    return features


def extract_fundamental_features(overview_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract fundamental features from company overview

    Args:
        overview_data: Response from get_company_overview()

    Returns:
        Dictionary with fundamental features
    """
    features = {}

    def safe_float(value: str, default: float = 0.0) -> float:
        """Safely convert string to float"""
        try:
            if value in ["None", "", "-"]:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    # P/E ratio (normalized)
    pe = safe_float(overview_data.get("PERatio"))
    features["pe_ratio"] = min(pe / 50.0, 1.0) if pe > 0 else 0.0  # Normalize to 0-1

    # P/B ratio
    pb = safe_float(overview_data.get("PriceToBookRatio"))
    features["pb_ratio"] = min(pb / 10.0, 1.0) if pb > 0 else 0.0

    # Dividend yield
    div_yield = safe_float(overview_data.get("DividendYield"))
    features["dividend_yield"] = div_yield

    # Profit margin
    profit_margin = safe_float(overview_data.get("ProfitMargin"))
    features["profit_margin"] = profit_margin

    # ROE
    roe = safe_float(overview_data.get("ReturnOnEquityTTM"))
    features["roe"] = roe

    # Beta (volatility vs market)
    beta = safe_float(overview_data.get("Beta"), 1.0)
    features["beta"] = beta

    # Earnings growth
    earnings_growth = safe_float(overview_data.get("QuarterlyEarningsGrowthYOY"))
    features["earnings_growth"] = earnings_growth

    # Revenue growth
    revenue_growth = safe_float(overview_data.get("QuarterlyRevenueGrowthYOY"))
    features["revenue_growth"] = revenue_growth

    return features


if __name__ == "__main__":
    # Test the client
    print("Testing Alpha Vantage API client...")

    client = AlphaVantageClient()

    # Test 1: News sentiment
    print("\nTest 1: News Sentiment")
    try:
        sentiment = client.get_news_sentiment("AAPL", limit=5)
        print(f"✅ Retrieved {len(sentiment.get('feed', []))} news articles")

        # Extract features
        features = extract_news_sentiment_features(sentiment, "AAPL")
        print(f"   f81 (sentiment): {features['f81']:.3f}")
        print(f"   f82 (volume): {features['f82']:.3f}")
        print(f"   f90 (consensus): {features['f90']:.3f}")
    except Exception as e:
        print(f"❌ News sentiment test failed: {e}")

    # Test 2: Company overview
    print("\nTest 2: Company Overview")
    try:
        overview = client.get_company_overview("AAPL")
        print(f"✅ Retrieved overview for {overview.get('Name')}")

        # Extract features
        features = extract_fundamental_features(overview)
        print(f"   PE Ratio: {features.get('pe_ratio', 0):.3f}")
        print(f"   ROE: {features.get('roe', 0):.3f}")
        print(f"   Beta: {features.get('beta', 0):.3f}")
    except Exception as e:
        print(f"❌ Company overview test failed: {e}")

    print("\n✅ Alpha Vantage MCP client ready for integration")
