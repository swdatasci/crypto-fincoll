"""YFinance Minute Bar Fetcher - No Authentication Required"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class YFinanceMinuteFetcher:
    """Fetch 1-minute OHLCV bars using yfinance (free, no auth required)"""

    def fetch_minute_bars(self, symbol: str, start_date: str = None, end_date: str = None, interval: str = "1m", period: str = None):
        """
        Fetch minute-level OHLCV data

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h')

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]

        Note:
            yfinance limits minute data to last 30 days for 1m interval
            For historical data, must fetch in 7-day chunks
        """
        try:
            ticker = yf.Ticker(symbol)

            # Use period if provided (simpler and more reliable)
            if period:
                df = ticker.history(period=period, interval=interval)
            else:
                # For 1-minute data, yfinance has limits:
                # - Can only get last 7 days at a time
                # - Maximum 30 days of history total

                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)

                # Check if date range is within yfinance limits
                days_diff = (end_dt - start_dt).days

                if interval == "1m" and days_diff > 7:
                    print(f"  ⚠️  {symbol}: yfinance 1m data limited to 7 days. Fetching in chunks...")
                    all_data = []

                    # Fetch in 7-day chunks
                    current_start = start_dt
                    while current_start < end_dt:
                        chunk_end = min(current_start + timedelta(days=7), end_dt)

                        df_chunk = ticker.history(
                            start=current_start,
                            end=chunk_end,
                            interval=interval
                        )

                        if not df_chunk.empty:
                            all_data.append(df_chunk)

                        current_start = chunk_end

                    if not all_data:
                        print(f"  ⚠️  {symbol}: No data")
                        return None

                    df = pd.concat(all_data, axis=0)
                else:
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval
                    )

            if df.empty:
                print(f"  ⚠️  {symbol}: No data")
                return None

            # Standardize column names
            df = df.reset_index()
            df = df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns and 'Date' in df.columns:
                df['timestamp'] = df['Date']

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)
            print(f"  ✅ {symbol}: {len(df)} bars")
            return df

        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            return None


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING YFINANCE MINUTE DATA")
    print("="*70 + "\n")

    fetcher = YFinanceMinuteFetcher()

    # Test 1: Recent 5 days (1-minute data)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Test 1: Fetching AAPL 1-minute bars: {start} to {end}\n")
    df = fetcher.fetch_minute_bars('AAPL', start, end, interval="1m")

    if df is not None:
        print(f"\n📊 Sample (first 10 bars):\n{df.head(10)}")
        print(f"\n📊 Sample (last 10 bars):\n{df.tail(10)}")
        print(f"\n✅ Total: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Date range: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # Test 2: 5-minute bars (can get more history)
    print("\n" + "="*70)
    start_5min = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"Test 2: Fetching MSFT 5-minute bars: {start_5min} to {end}\n")
    df_5min = fetcher.fetch_minute_bars('MSFT', start_5min, end, interval="5m")

    if df_5min is not None:
        print(f"\n✅ Total: {len(df_5min)} bars from {df_5min['timestamp'].min()} to {df_5min['timestamp'].max()}")
        print(f"   Date range: {(df_5min['timestamp'].max() - df_5min['timestamp'].min()).days} days")
