"""
Fundamental History Tracker - Historical Time Series of Fundamentals

KEY INSIGHT: Fundamentals change over time (quarterly updates)!
We need to track WHEN they changed and HOW MUCH they changed.

User Quote: "fundamentals, ideally would be historically changing, but we don't
have change-dates... may need to add production-training-feedback loop"

What this does:
- Fetches quarterly historical fundamentals from yfinance
- Creates time series: [(date, P/E), (date, P/E), ...] for each metric
- Interpolates between quarters (linear or forward-fill)
- Adds delta features: "P/E increased 20% this quarter"
- Tracks earnings calendar to know when fundamentals update

Agent Delta - Phase 1 FEATURE INTEGRATION stream
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json


class FundamentalHistoryTracker:
    """
    Tracks historical changes in fundamental metrics.

    This is crucial because fundamentals don't change daily - they update
    quarterly (earnings releases). The model needs to know:
    1. What were the fundamentals at each point in time?
    2. How did they change quarter-over-quarter?
    3. When is the next earnings release? (high volatility period)
    """

    def __init__(self, cache_dir: Optional[str] = None, cache_ttl_hours: int = 168):
        """
        Initialize the fundamental history tracker.

        Args:
            cache_dir: Directory to store cached data (default: data/cache/fundamental_history)
            cache_ttl_hours: Hours before cache expires (default: 168 = 1 week)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'cache' / 'fundamental_history'
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"{symbol}_history.json"

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data exists and is still valid."""
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return False

        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_ttl

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load fundamental history from cache if valid."""
        if not self._is_cache_valid(symbol):
            return None

        try:
            cache_path = self._get_cache_path(symbol)
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Convert back to DataFrame
            df = pd.DataFrame(data['history'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            return df
        except Exception as e:
            print(f"Warning: Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save fundamental history to cache."""
        try:
            cache_path = self._get_cache_path(symbol)

            # Convert DataFrame to JSON-serializable format
            df_copy = df.reset_index()
            df_copy['date'] = df_copy['date'].astype(str)

            data = {
                'symbol': symbol,
                'last_updated': datetime.now().isoformat(),
                'history': df_copy.to_dict(orient='records')
            }

            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save cache for {symbol}: {e}")

    def get_historical_fundamentals(self, symbol: str,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical quarterly fundamentals for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: [date (index), pe_ratio, pb_ratio, roe, ...]
            One row per quarter, with forward-fill interpolation for daily usage
        """
        # Check cache first
        if use_cache:
            cached_df = self._load_from_cache(symbol)
            if cached_df is not None:
                print(f"Loaded {symbol} fundamental history from cache")
                # Filter by date range if specified
                if start_date:
                    cached_df = cached_df[cached_df.index >= start_date]
                if end_date:
                    cached_df = cached_df[cached_df.index <= end_date]
                return cached_df

        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=5*365)  # 5 years

        print(f"Fetching historical fundamentals for {symbol}...")

        try:
            ticker = yf.Ticker(symbol)

            # Fetch quarterly financials
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow

            # Get quarterly dates (columns are dates)
            if quarterly_financials is None or quarterly_financials.empty:
                print(f"No quarterly data available for {symbol}")
                return pd.DataFrame()

            dates = quarterly_financials.columns

            # Build historical time series
            history = []

            for date in dates:
                # Extract metrics for this quarter
                metrics = self._extract_quarterly_metrics(
                    quarterly_financials,
                    quarterly_balance,
                    quarterly_cashflow,
                    date
                )

                metrics['date'] = pd.to_datetime(date)
                history.append(metrics)

            # Create DataFrame
            df = pd.DataFrame(history)
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)

            # Calculate quarter-over-quarter deltas
            df = self._add_delta_features(df)

            # Save to cache
            if use_cache:
                self._save_to_cache(symbol, df)

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            return df

        except Exception as e:
            print(f"Error fetching historical fundamentals for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _extract_quarterly_metrics(self,
                                   financials: pd.DataFrame,
                                   balance_sheet: pd.DataFrame,
                                   cashflow: pd.DataFrame,
                                   date: pd.Timestamp) -> Dict[str, Optional[float]]:
        """
        Extract fundamental metrics for a specific quarter.

        Args:
            financials: Quarterly financials DataFrame
            balance_sheet: Quarterly balance sheet DataFrame
            cashflow: Quarterly cashflow DataFrame
            date: Quarter date

        Returns:
            Dictionary with fundamental metrics for this quarter
        """
        metrics = {}

        # Helper function to safely get value
        def get_value(df, row_name):
            try:
                if df is not None and row_name in df.index and date in df.columns:
                    val = df.loc[row_name, date]
                    return float(val) if pd.notna(val) else None
            except Exception:
                pass
            return None

        # Income statement metrics
        metrics['total_revenue'] = get_value(financials, 'Total Revenue')
        metrics['net_income'] = get_value(financials, 'Net Income')
        metrics['operating_income'] = get_value(financials, 'Operating Income')
        metrics['gross_profit'] = get_value(financials, 'Gross Profit')

        # Balance sheet metrics
        metrics['total_assets'] = get_value(balance_sheet, 'Total Assets')
        metrics['total_equity'] = get_value(balance_sheet, 'Total Equity Gross Minority Interest')
        metrics['total_debt'] = get_value(balance_sheet, 'Total Debt')
        metrics['current_assets'] = get_value(balance_sheet, 'Current Assets')
        metrics['current_liabilities'] = get_value(balance_sheet, 'Current Liabilities')
        metrics['inventory'] = get_value(balance_sheet, 'Inventory')

        # Cash flow metrics
        metrics['operating_cashflow'] = get_value(cashflow, 'Operating Cash Flow')
        metrics['capex'] = get_value(cashflow, 'Capital Expenditure')
        metrics['free_cashflow'] = None
        if metrics['operating_cashflow'] and metrics['capex']:
            metrics['free_cashflow'] = metrics['operating_cashflow'] - abs(metrics['capex'])

        # Calculate derived ratios
        # ROE = Net Income / Total Equity
        if metrics['net_income'] and metrics['total_equity']:
            metrics['roe'] = (metrics['net_income'] / metrics['total_equity']) * 100

        # ROA = Net Income / Total Assets
        if metrics['net_income'] and metrics['total_assets']:
            metrics['roa'] = (metrics['net_income'] / metrics['total_assets']) * 100

        # Profit Margin = Net Income / Total Revenue
        if metrics['net_income'] and metrics['total_revenue']:
            metrics['profit_margin'] = (metrics['net_income'] / metrics['total_revenue']) * 100

        # Debt-to-Equity
        if metrics['total_debt'] and metrics['total_equity']:
            metrics['debt_to_equity'] = metrics['total_debt'] / metrics['total_equity']

        # Current Ratio
        if metrics['current_assets'] and metrics['current_liabilities']:
            metrics['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']

        # Quick Ratio
        if metrics['current_assets'] and metrics['inventory'] and metrics['current_liabilities']:
            quick_assets = metrics['current_assets'] - metrics['inventory']
            metrics['quick_ratio'] = quick_assets / metrics['current_liabilities']

        return metrics

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quarter-over-quarter delta features.

        These show HOW MUCH fundamentals changed, which is often more important
        than the absolute values.

        Args:
            df: DataFrame with quarterly fundamentals

        Returns:
            DataFrame with added delta columns (e.g., 'roe_delta_qoq')
        """
        result = df.copy()

        # Metrics to calculate deltas for
        delta_metrics = [
            'roe', 'roa', 'profit_margin', 'debt_to_equity',
            'current_ratio', 'quick_ratio', 'total_revenue',
            'net_income', 'operating_income', 'free_cashflow'
        ]

        for metric in delta_metrics:
            if metric in result.columns:
                # Absolute change
                result[f'{metric}_delta_qoq'] = result[metric].diff()

                # Percentage change
                result[f'{metric}_pct_change_qoq'] = result[metric].pct_change() * 100

        return result

    def interpolate_to_daily(self, df: pd.DataFrame,
                            start_date: datetime,
                            end_date: datetime,
                            method: str = 'ffill') -> pd.DataFrame:
        """
        Interpolate quarterly data to daily frequency.

        This is needed for training, since we need fundamental values
        aligned with daily price data.

        Args:
            df: DataFrame with quarterly fundamentals
            start_date: Start date for daily data
            end_date: End date for daily data
            method: Interpolation method ('ffill' = forward fill, 'linear' = linear interpolation)

        Returns:
            DataFrame with daily fundamentals (forward-filled from quarters)
        """
        if df.empty:
            return df

        # Create daily date range
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex to daily frequency
        df_daily = df.reindex(daily_dates)

        # Interpolate based on method
        if method == 'ffill':
            # Forward fill - use last known quarterly value
            df_daily = df_daily.fillna(method='ffill')
        elif method == 'linear':
            # Linear interpolation between quarters
            df_daily = df_daily.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return df_daily

    def get_next_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the next earnings release date for a symbol.

        Earnings releases are high-volatility events where fundamentals update.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Next earnings date or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is not None and 'Earnings Date' in calendar:
                earnings_date = calendar['Earnings Date']

                # Handle different return types from yfinance
                if isinstance(earnings_date, list):
                    # Take the first date if it's a list
                    earnings_date = earnings_date[0] if len(earnings_date) > 0 else None
                elif isinstance(earnings_date, pd.Series):
                    earnings_date = earnings_date.iloc[0]

                if earnings_date is None:
                    return None

                # Convert to datetime if needed
                if isinstance(earnings_date, str):
                    earnings_date = pd.to_datetime(earnings_date)
                elif hasattr(earnings_date, 'to_pydatetime'):
                    # Convert pandas Timestamp to datetime
                    earnings_date = earnings_date.to_pydatetime()
                elif isinstance(earnings_date, datetime):
                    # Already datetime, keep as is
                    pass
                else:
                    # Convert date to datetime if needed
                    from datetime import date
                    if isinstance(earnings_date, date):
                        earnings_date = datetime.combine(earnings_date, datetime.min.time())

                return earnings_date

            return None

        except Exception as e:
            print(f"Error fetching earnings date for {symbol}: {e}")
            return None

    def get_days_until_earnings(self, symbol: str) -> Optional[int]:
        """
        Get number of days until next earnings release.

        This is useful for risk management - model predictions are less reliable
        right before earnings.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Number of days until earnings or None if not available
        """
        earnings_date = self.get_next_earnings_date(symbol)

        if earnings_date is None:
            return None

        days_until = (earnings_date - datetime.now()).days
        return days_until

    def get_fundamental_at_date(self, df: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        Get fundamental values at a specific date (with forward fill).

        Args:
            df: DataFrame with quarterly fundamentals
            date: Target date

        Returns:
            Dictionary with fundamental values at that date
        """
        if df.empty:
            return {}

        # Find the most recent quarter before or on this date
        past_quarters = df[df.index <= date]

        if past_quarters.empty:
            # No data before this date - use first available
            return df.iloc[0].to_dict()

        # Return most recent quarter
        return past_quarters.iloc[-1].to_dict()


if __name__ == "__main__":
    # Quick test
    tracker = FundamentalHistoryTracker()

    # Test on AAPL
    symbol = "AAPL"
    print(f"Testing fundamental history for {symbol}...")

    # Get 2 years of quarterly data
    start_date = datetime.now() - timedelta(days=2*365)
    df = tracker.get_historical_fundamentals(symbol, start_date=start_date)

    print(f"\nQuarterly Data Shape: {df.shape}")
    print(f"Available Columns: {list(df.columns)}")
    print(f"\nLast 3 Quarters:")
    print(df[['roe', 'roa', 'profit_margin', 'roe_delta_qoq']].tail(3))

    # Test interpolation to daily
    print(f"\nInterpolating to daily frequency...")
    end_date = datetime.now()
    start_daily = end_date - timedelta(days=90)  # Last 90 days
    df_daily = tracker.interpolate_to_daily(df, start_daily, end_date)

    print(f"Daily Data Shape: {df_daily.shape}")
    print(f"\nLast 5 Days:")
    print(df_daily[['roe', 'roa', 'profit_margin']].tail(5))

    # Test earnings date
    print(f"\nNext earnings date:")
    earnings_date = tracker.get_next_earnings_date(symbol)
    print(f"  Date: {earnings_date}")

    days_until = tracker.get_days_until_earnings(symbol)
    if days_until:
        print(f"  Days until: {days_until}")
