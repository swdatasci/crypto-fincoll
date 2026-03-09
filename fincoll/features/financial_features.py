"""
Financial Feature Calculator for FinVec V2
Calculates mathematical derivatives and technical indicators for crash detection
"""

import pandas as pd
import numpy as np
from typing import Tuple


class FinancialFeatureCalculator:
    """
    Calculate advanced financial features for FinVec v2:
    - Mathematical derivatives (velocity, acceleration, jerk)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Volume analysis (spikes, relative volume)
    - Volatility metrics (VVIX, volatility regimes)
    """

    # ========================================
    # Mathematical Derivatives
    # ========================================

    @staticmethod
    def calculate_velocity(df: pd.DataFrame, window: int = 1, price_col: str = 'close') -> pd.Series:
        """
        Calculate price velocity (first derivative)

        Args:
            df: DataFrame with OHLCV data
            window: Window for calculating change (default: 1 day)
            price_col: Column to use for calculation

        Returns:
            Series of velocity values (% change per day)
        """
        return df[price_col].pct_change(window) * 100  # Convert to percentage

    @staticmethod
    def calculate_acceleration(df: pd.DataFrame, window: int = 1, price_col: str = 'close') -> pd.Series:
        """
        Calculate price acceleration (second derivative)

        Args:
            df: DataFrame with OHLCV data
            window: Window for calculating change
            price_col: Column to use for calculation

        Returns:
            Series of acceleration values (%/day²)
        """
        velocity = FinancialFeatureCalculator.calculate_velocity(df, window, price_col)
        return velocity.diff(window)  # Acceleration = change in velocity

    @staticmethod
    def calculate_jerk(df: pd.DataFrame, window: int = 1, price_col: str = 'close') -> pd.Series:
        """
        Calculate price jerk (third derivative) - detects regime shifts

        Args:
            df: DataFrame with OHLCV data
            window: Window for calculating change
            price_col: Column to use for calculation

        Returns:
            Series of jerk values (%/day³)
        """
        acceleration = FinancialFeatureCalculator.calculate_acceleration(df, window, price_col)
        return acceleration.diff(window)  # Jerk = change in acceleration

    # ========================================
    # Technical Indicators
    # ========================================

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Thresholds:
        - RSI < 30: Oversold (potential bounce or crash acceleration)
        - RSI > 70: Overbought (potential pullback or bubble)

        Args:
            df: DataFrame with OHLCV data
            period: RSI period (default: 14)
            price_col: Column to use for calculation

        Returns:
            Series of RSI values (0-100)
        """
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD)

        MACD crossovers:
        - MACD > Signal: Bullish (uptrend)
        - MACD < Signal: Bearish (downtrend, crash warning)

        Divergences:
        - Price up, MACD down: Top forming (crash risk)
        - Price down, MACD up: Bottom forming (reversal)

        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            price_col: Column to use for calculation

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
        price_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Position interpretation:
        - Price > Upper band (+2σ): Overbought, crash risk
        - Price < Lower band (-2σ): Oversold, bounce expected
        - Bands narrowing: Low volatility, breakout imminent

        Args:
            df: DataFrame with OHLCV data
            period: Moving average period (default: 20)
            std: Standard deviation multiplier (default: 2.0)
            price_col: Column to use for calculation

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = df[price_col].rolling(window=period).mean()
        std_dev = df[price_col].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_bb_position(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        Calculate relative position within Bollinger Bands

        Returns:
            Series of position values:
            -  0.0 = at middle band
            - +1.0 = at upper band (+2σ)
            - -1.0 = at lower band (-2σ)
            - >+1.0 = above upper band (extreme overbought)
        """
        upper, middle, lower = FinancialFeatureCalculator.calculate_bollinger_bands(
            df, period, std, price_col
        )

        # Normalize position to [-1, 1] scale
        band_width = upper - lower
        position = (df[price_col] - middle) / (band_width / 2)

        return position

    # ========================================
    # Volume Analysis
    # ========================================

    @staticmethod
    def calculate_volume_spike(
        df: pd.DataFrame,
        period: int = 20,
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Calculate volume spike ratio

        Thresholds:
        - > 3x: Extreme spike (panic selling or FOMO buying)
        - > 2x: Unusual activity
        - < 0.5x: Quiet market (potential trap)

        Args:
            df: DataFrame with OHLCV data
            period: Period for moving average (default: 20)
            volume_col: Column to use for calculation

        Returns:
            Series of volume ratios (current / average)
        """
        volume_ma = df[volume_col].rolling(window=period).mean()
        return df[volume_col] / volume_ma

    # ========================================
    # Volatility Analysis
    # ========================================

    @staticmethod
    def calculate_volatility(
        df: pd.DataFrame,
        window: int = 20,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns)

        Args:
            df: DataFrame with OHLCV data
            window: Rolling window (default: 20)
            price_col: Column to use for calculation

        Returns:
            Series of volatility values (annualized %)
        """
        returns = df[price_col].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized

        return volatility

    @staticmethod
    def calculate_vvix(
        df: pd.DataFrame,
        vol_window: int = 20,
        vvix_window: int = 5,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        Calculate VVIX (volatility of volatility)

        High VVIX indicates:
        - Fear is accelerating
        - Market uncertainty rising
        - Crash risk elevated

        Args:
            df: DataFrame with OHLCV data
            vol_window: Window for volatility calculation
            vvix_window: Window for VVIX calculation
            price_col: Column to use for calculation

        Returns:
            Series of VVIX values
        """
        volatility = FinancialFeatureCalculator.calculate_volatility(df, vol_window, price_col)
        vvix = volatility.rolling(window=vvix_window).std()

        return vvix

    # ========================================
    # Composite Features
    # ========================================

    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all v2 features at once

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with original data plus all calculated features
        """
        result = df.copy()

        # Mathematical derivatives
        result['velocity'] = FinancialFeatureCalculator.calculate_velocity(result)
        result['acceleration'] = FinancialFeatureCalculator.calculate_acceleration(result)
        result['jerk'] = FinancialFeatureCalculator.calculate_jerk(result)

        # Technical indicators
        result['rsi'] = FinancialFeatureCalculator.calculate_rsi(result)

        macd, signal, histogram = FinancialFeatureCalculator.calculate_macd(result)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram

        upper, middle, lower = FinancialFeatureCalculator.calculate_bollinger_bands(result)
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        result['bb_position'] = FinancialFeatureCalculator.calculate_bb_position(result)

        # Volume analysis
        result['volume_spike'] = FinancialFeatureCalculator.calculate_volume_spike(result)

        # Volatility analysis
        result['volatility'] = FinancialFeatureCalculator.calculate_volatility(result)
        result['vvix'] = FinancialFeatureCalculator.calculate_vvix(result)

        return result


if __name__ == "__main__":
    # Example usage and testing
    import yfinance as yf

    print("Testing FinancialFeatureCalculator...")

    # Download sample data
    df = yf.Ticker('AAPL').history(period='1y')

    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # Calculate all features
    calc = FinancialFeatureCalculator()
    features = calc.calculate_all_features(df)

    print(f"\n✅ Calculated {len(features.columns)} total columns")
    print(f"\nNew features added:")
    new_cols = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in new_cols:
        print(f"  - {col}")

    print(f"\nLast 5 rows of features:")
    print(features[new_cols].tail())

    print(f"\n📊 Feature statistics (last value):")
    print(f"  Velocity: {features['velocity'].iloc[-1]:.2f}%/day")
    print(f"  Acceleration: {features['acceleration'].iloc[-1]:.2f}%/day²")
    print(f"  RSI: {features['rsi'].iloc[-1]:.1f}")
    print(f"  MACD: {features['macd'].iloc[-1]:.2f}")
    print(f"  BB Position: {features['bb_position'].iloc[-1]:.2f}σ")
    print(f"  Volume Spike: {features['volume_spike'].iloc[-1]:.2f}x")
    print(f"  Volatility: {features['volatility'].iloc[-1]:.1f}%")
