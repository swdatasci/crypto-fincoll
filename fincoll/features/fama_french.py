"""
Fama-French Factor Data Fetcher and Calculator

Fetches factor data from Kenneth French Data Library and calculates
factor exposures (betas) for individual stocks.

Factors:
- SMB: Small Minus Big (size factor)
- HML: High Minus Low (value factor)
- RMW: Robust Minus Weak (profitability factor)
- CMA: Conservative Minus Aggressive (investment factor)
- UMD: Up Minus Down (momentum factor)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import requests
import io

logger = logging.getLogger(__name__)


class FamaFrenchFactors:
    """Fetches and caches Fama-French factor data"""

    # Kenneth French Data Library URLs
    FF5_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def __init__(self, cache_dir: str = "/tmp/fama_french_cache"):
        """
        Initialize Fama-French data fetcher

        Args:
            cache_dir: Directory to cache downloaded factor data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._ff5_data = None
        self._mom_data = None
        self._last_fetch = None
        self._cache_ttl = timedelta(days=1)  # Refresh daily

    def fetch_factors(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch all Fama-French factors (5-factor + momentum)

        Args:
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with columns: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
            Index: datetime dates
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached Fama-French factor data")
            return self._merge_factors()

        try:
            # Fetch 5-factor data
            logger.info("Fetching Fama-French 5-factor data...")
            self._ff5_data = self._fetch_ff5_factors()

            # Fetch momentum data
            logger.info("Fetching momentum factor data...")
            self._mom_data = self._fetch_momentum_factor()

            self._last_fetch = datetime.now()

            # Save to cache
            self._save_cache()

            return self._merge_factors()

        except Exception as e:
            logger.error(f"Failed to fetch Fama-French factors: {e}")
            # Try to load from disk cache
            return self._load_cache()

    def _fetch_ff5_factors(self) -> pd.DataFrame:
        """Fetch 5-factor model data (Mkt-RF, SMB, HML, RMW, CMA, RF)"""
        try:
            # Download ZIP file
            response = requests.get(self.FF5_DAILY_URL, timeout=30)
            response.raise_for_status()

            # Read CSV from ZIP
            import zipfile
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Skip header lines, read until annual factors start
                    content = f.read().decode('utf-8')

            # Parse CSV (data starts after first blank line)
            lines = content.split('\n')
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == '':
                    data_start = i + 1
                    break

            # Find where annual data starts (marked by blank line after daily data)
            data_end = len(lines)
            for i in range(data_start, len(lines)):
                if lines[i].strip() == '':
                    data_end = i
                    break

            # Read daily data
            daily_data = '\n'.join(lines[data_start:data_end])
            df = pd.read_csv(io.StringIO(daily_data))

            # Convert date column (format: YYYYMMDD)
            df['Date'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d')
            df = df.set_index('Date')

            # Convert percentages to decimals (data is in percentage points)
            factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            for col in factor_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float) / 100.0

            logger.info(f"Fetched {len(df)} days of 5-factor data")
            return df[factor_cols]

        except Exception as e:
            logger.error(f"Failed to fetch 5-factor data: {e}")
            raise

    def _fetch_momentum_factor(self) -> pd.DataFrame:
        """Fetch momentum factor (UMD/Mom)"""
        try:
            # Download ZIP file
            response = requests.get(self.MOM_DAILY_URL, timeout=30)
            response.raise_for_status()

            # Read CSV from ZIP
            import zipfile
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')

            # Parse CSV
            lines = content.split('\n')
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == '':
                    data_start = i + 1
                    break

            data_end = len(lines)
            for i in range(data_start, len(lines)):
                if lines[i].strip() == '':
                    data_end = i
                    break

            daily_data = '\n'.join(lines[data_start:data_end])
            df = pd.read_csv(io.StringIO(daily_data))

            # Convert date
            df['Date'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d')
            df = df.set_index('Date')

            # Convert percentage to decimal
            df['Mom'] = df.iloc[:, 0].astype(float) / 100.0

            logger.info(f"Fetched {len(df)} days of momentum data")
            return df[['Mom']]

        except Exception as e:
            logger.error(f"Failed to fetch momentum data: {e}")
            raise

    def _merge_factors(self) -> pd.DataFrame:
        """Merge 5-factor and momentum data"""
        if self._ff5_data is None or self._mom_data is None:
            raise ValueError("Factor data not loaded")

        # Merge on date index
        merged = self._ff5_data.join(self._mom_data, how='inner')
        return merged

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if self._ff5_data is None or self._mom_data is None:
            return False
        if self._last_fetch is None:
            return False
        if datetime.now() - self._last_fetch > self._cache_ttl:
            return False
        return True

    def _save_cache(self):
        """Save factor data to disk cache"""
        try:
            if self._ff5_data is not None:
                self._ff5_data.to_parquet(self.cache_dir / "ff5_daily.parquet")
            if self._mom_data is not None:
                self._mom_data.to_parquet(self.cache_dir / "mom_daily.parquet")

            # Save metadata
            with open(self.cache_dir / "last_fetch.txt", 'w') as f:
                f.write(datetime.now().isoformat())

            logger.info("Saved Fama-French data to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load factor data from disk cache"""
        try:
            ff5_path = self.cache_dir / "ff5_daily.parquet"
            mom_path = self.cache_dir / "mom_daily.parquet"

            if ff5_path.exists() and mom_path.exists():
                self._ff5_data = pd.read_parquet(ff5_path)
                self._mom_data = pd.read_parquet(mom_path)

                # Check last fetch time
                last_fetch_path = self.cache_dir / "last_fetch.txt"
                if last_fetch_path.exists():
                    with open(last_fetch_path) as f:
                        self._last_fetch = datetime.fromisoformat(f.read().strip())

                logger.info("Loaded Fama-French data from cache")
                return self._merge_factors()
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

        return None

    def calculate_factor_exposures(
        self,
        stock_returns: pd.Series,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        Calculate factor exposures (betas) for a stock using rolling regression

        Args:
            stock_returns: Series of stock returns (index: dates)
            lookback_days: Number of days for rolling regression

        Returns:
            Dictionary with factor betas: {'SMB': 0.5, 'HML': -0.2, ...}
        """
        try:
            # Get factor data
            factors = self.fetch_factors()

            # Align stock returns with factor data
            common_dates = stock_returns.index.intersection(factors.index)
            if len(common_dates) < lookback_days:
                logger.warning(f"Insufficient data for factor regression: {len(common_dates)} < {lookback_days}")
                return self._zero_exposures()

            # Get recent data
            recent_dates = common_dates[-lookback_days:]
            y = stock_returns.loc[recent_dates].values
            X = factors.loc[recent_dates, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].values
            rf = factors.loc[recent_dates, 'RF'].values

            # Excess returns (stock return - risk-free rate)
            y_excess = y - rf

            # Run regression: y_excess = alpha + beta * factors
            # Using numpy lstsq for speed
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            betas, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y_excess, rcond=None)

            # Extract factor betas (skip intercept)
            return {
                'market': betas[1],  # Mkt-RF beta
                'SMB': betas[2],     # Size factor
                'HML': betas[3],     # Value factor
                'RMW': betas[4],     # Profitability factor
                'CMA': betas[5],     # Investment factor
                'Mom': betas[6],     # Momentum factor
            }

        except Exception as e:
            logger.error(f"Failed to calculate factor exposures: {e}")
            return self._zero_exposures()

    def _zero_exposures(self) -> Dict[str, float]:
        """Return zero exposures (fallback)"""
        return {
            'market': 0.0,
            'SMB': 0.0,
            'HML': 0.0,
            'RMW': 0.0,
            'CMA': 0.0,
            'Mom': 0.0,
        }


# Global instance for caching
_ff_instance = None

def get_fama_french_factors() -> FamaFrenchFactors:
    """Get global Fama-French factor instance (cached)"""
    global _ff_instance
    if _ff_instance is None:
        _ff_instance = FamaFrenchFactors()
    return _ff_instance
