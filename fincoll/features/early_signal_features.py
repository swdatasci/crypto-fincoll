"""
Early Signal Features for Peak/Valley Prediction

These features are designed to give the model EARLY indicators of:
1. Momentum building (price acceleration)
2. Volume surges (volume acceleration)
3. Velocity changes (jerk - 3rd derivative)
4. Pattern breakouts (distance from support/resistance)

The goal: Predict peaks/valleys BEFORE they happen, not after.
"""

from typing import Optional

import numpy as np
import pandas as pd


def extract_early_signal_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 30D early signal features focused on momentum/acceleration/jerk.

    Returns:
        30D numpy array of float32 features
    """
    features = []

    if len(df) < 5:
        return np.zeros(30, dtype=np.float32)

    prices = df["close"].values
    volumes = df["volume"].values
    highs = df["high"].values
    lows = df["low"].values

    # === PRICE DERIVATIVES (10D) ===

    # 1-3: Velocity at different scales (1st derivative)
    for period in [3, 5, 10]:
        if len(prices) >= period + 1:
            velocity = (prices[-1] - prices[-period - 1]) / period
            features.append(velocity / prices[-1])  # Normalized
        else:
            features.append(0.0)

    # 4-6: Acceleration (2nd derivative) - KEY for early signals
    for period in [3, 5, 10]:
        if len(prices) >= period + 2:
            # Acceleration = change in velocity
            vel_now = (prices[-1] - prices[-period]) / period
            vel_before = (prices[-period] - prices[-period * 2]) / period
            accel = vel_now - vel_before
            features.append(accel / prices[-1])
        else:
            features.append(0.0)

    # 7-8: Jerk (3rd derivative) - VERY early signal
    if len(prices) >= 8:
        # Jerk = change in acceleration
        accel_now = prices[-1] - 2 * prices[-3] + prices[-5]
        accel_before = prices[-3] - 2 * prices[-5] + prices[-7]
        jerk = accel_now - accel_before
        features.append(jerk / prices[-1])
    else:
        features.append(0.0)

    if len(prices) >= 12:
        accel_now = prices[-1] - 2 * prices[-5] + prices[-9]
        accel_before = prices[-5] - 2 * prices[-9] + prices[-13]
        jerk = accel_now - accel_before
        features.append(jerk / prices[-1])
    else:
        features.append(0.0)

    # 9-10: Acceleration trend (is acceleration increasing?)
    for period in [5, 10]:
        if len(prices) >= period + 2:  # Need period + 2 bars minimum
            accels = []
            # Calculate accelerations for last 'period' bars
            for i in range(period):
                idx = len(prices) - 1 - i  # Start from end
                if idx >= 2:  # Need at least 3 points for acceleration
                    a = prices[idx] - 2 * prices[idx - 1] + prices[idx - 2]
                    accels.append(a)

            if len(accels) >= 3:
                # Linear fit slope of accelerations
                accels_array = np.array(accels)
                x = np.arange(len(accels_array))
                slope = np.polyfit(x, accels_array, 1)[0]
                features.append(slope / prices[-1] if prices[-1] != 0 else 0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)

    # === VOLUME DERIVATIVES (8D) ===

    # 11-12: Volume velocity
    for period in [3, 5]:
        if len(volumes) >= period + 1:
            vol_vel = (volumes[-1] - volumes[-period - 1]) / period
            avg_vol = volumes[-(period + 1) :].mean()
            features.append(vol_vel / avg_vol if avg_vol > 0 else 0.0)
        else:
            features.append(0.0)

    # 13-14: Volume acceleration (SURGE detector)
    for period in [3, 5]:
        if len(volumes) >= period * 2:
            vol_vel_now = (volumes[-1] - volumes[-period]) / period
            vol_vel_before = (volumes[-period] - volumes[-period * 2]) / period
            vol_accel = vol_vel_now - vol_vel_before
            avg_vol = volumes[-period:].mean()
            features.append(vol_accel / avg_vol if avg_vol > 0 else 0.0)
        else:
            features.append(0.0)

    # 15: Volume jerk (early surge detection)
    if len(volumes) >= 8:
        accel_now = volumes[-1] - 2 * volumes[-3] + volumes[-5]
        accel_before = volumes[-3] - 2 * volumes[-5] + volumes[-7]
        jerk = accel_now - accel_before
        avg_vol = volumes[-5:].mean()
        features.append(jerk / avg_vol if avg_vol > 0 else 0.0)
    else:
        features.append(0.0)

    # 16-18: Volume ratio acceleration (relative volume building)
    for period in [5, 10, 20]:
        if len(volumes) >= period * 2:
            ratio_now = (
                volumes[-1] / volumes[-(period + 1) :].mean()
                if volumes[-(period + 1) :].mean() > 0
                else 1.0
            )
            ratio_before = (
                volumes[-period] / volumes[-(period * 2 + 1) : -period].mean()
                if volumes[-(period * 2 + 1) : -period].mean() > 0
                else 1.0
            )
            ratio_accel = ratio_now - ratio_before
            features.append(ratio_accel)
        else:
            features.append(0.0)

    # === PRICE POSITION & MOMENTUM (12D) ===

    # 19-21: Distance from recent highs (potential breakout)
    for period in [5, 10, 20]:
        if len(highs) >= period:
            recent_high = highs[-period:].max()
            distance = (prices[-1] - recent_high) / recent_high
            features.append(distance)
        else:
            features.append(0.0)

    # 22-24: Distance from recent lows (potential breakdown)
    for period in [5, 10, 20]:
        if len(lows) >= period:
            recent_low = lows[-period:].min()
            distance = (prices[-1] - recent_low) / recent_low
            features.append(distance)
        else:
            features.append(0.0)

    # 25-27: Momentum strength (price vs moving average, with acceleration)
    for period in [5, 10, 20]:
        if len(prices) >= period + 1:
            ma = prices[-(period + 1) : -1].mean()
            current_distance = (prices[-1] - ma) / ma if ma > 0 else 0.0

            if len(prices) >= period + 3:
                ma_before = prices[-(period + 3) : -3].mean()
                previous_distance = (
                    (prices[-3] - ma_before) / ma_before if ma_before > 0 else 0.0
                )
                momentum_accel = current_distance - previous_distance
                features.append(momentum_accel)
            else:
                features.append(current_distance)
        else:
            features.append(0.0)

    # 28-30: Price volatility acceleration (volatility building = potential move)
    for period in [5, 10, 20]:
        if len(prices) >= period * 2 + 1:
            # Calculate returns: (price[i+1] - price[i]) / price[i]
            recent_prices = prices[-period:]
            if len(recent_prices) >= 2:
                recent_returns = np.diff(recent_prices) / recent_prices[:-1]
            else:
                recent_returns = np.array([])

            older_prices = prices[-period * 2 : -period]
            if len(older_prices) >= 2:
                older_returns = np.diff(older_prices) / older_prices[:-1]
            else:
                older_returns = np.array([])

            vol_recent = np.std(recent_returns) if len(recent_returns) > 0 else 0.0
            vol_older = np.std(older_returns) if len(older_returns) > 0 else 0.0
            vol_accel = vol_recent - vol_older
            features.append(vol_accel)
        else:
            features.append(0.0)

    # Ensure exactly 30 features
    features = features[:30]
    while len(features) < 30:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_feature_names() -> list:
    """Return descriptive names for all 30 features"""
    return [
        # Price derivatives (10)
        "price_velocity_3bar",
        "price_velocity_5bar",
        "price_velocity_10bar",
        "price_accel_3bar",
        "price_accel_5bar",
        "price_accel_10bar",
        "price_jerk_8bar",
        "price_jerk_12bar",
        "accel_trend_5bar",
        "accel_trend_10bar",
        # Volume derivatives (8)
        "volume_velocity_3bar",
        "volume_velocity_5bar",
        "volume_accel_3bar",
        "volume_accel_5bar",
        "volume_jerk_8bar",
        "volume_ratio_accel_5bar",
        "volume_ratio_accel_10bar",
        "volume_ratio_accel_20bar",
        # Price position (12)
        "distance_from_high_5bar",
        "distance_from_high_10bar",
        "distance_from_high_20bar",
        "distance_from_low_5bar",
        "distance_from_low_10bar",
        "distance_from_low_20bar",
        "momentum_accel_vs_ma5",
        "momentum_accel_vs_ma10",
        "momentum_accel_vs_ma20",
        "volatility_accel_5bar",
        "volatility_accel_10bar",
        "volatility_accel_20bar",
    ]
