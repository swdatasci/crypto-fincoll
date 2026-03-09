"""
Velocity Target Computation for FinColl Training Orchestration

This module computes velocity targets from OHLCV data for training.
It wraps FinVec's velocity target computation logic with a FinColl-friendly interface.

Velocity targets represent the best profit opportunity (% return per bar) at each timeframe:
- For LONG: best peak return / bars to peak
- For SHORT: best valley return / bars to valley

This is used by the orchestration layer when preparing datasets for FinVec training.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_velocity_targets(
    bars: Dict[str, pd.DataFrame], timeframes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute velocity targets for all timeframes.

    This is the PRIMARY function called by FinColl orchestration to prepare
    training targets for FinVec velocity models.

    Args:
        bars: Dictionary mapping timeframe -> DataFrame
              e.g., {'1m': df_1min, '15m': df_15min, '1h': df_1hour, '1d': df_daily}
              Each DataFrame must have columns: open, high, low, close, volume
        timeframes: List of timeframe names to compute (default: ['1m', '15m', '1h', '1d'])

    Returns:
        DataFrame with velocity targets aligned to finest timeframe (1m).
        Columns:
        - {tf}_long_velocity: Best long velocity (% return per bar)
        - {tf}_long_bars: Number of bars until peak
        - {tf}_short_velocity: Best short velocity (% return per bar, negative)
        - {tf}_short_bars: Number of bars until valley

        Example output columns:
        - 1m_long_velocity, 1m_long_bars, 1m_short_velocity, 1m_short_bars
        - 15m_long_velocity, 15m_long_bars, 15m_short_velocity, 15m_short_bars
        - 1h_long_velocity, 1h_long_bars, 1h_short_velocity, 1h_short_bars
        - 1d_long_velocity, 1d_long_bars, 1d_short_velocity, 1d_short_bars

    Example:
        >>> # Fetch multi-timeframe data
        >>> from fincoll.data import fetch_multi_timeframe
        >>> bars = fetch_multi_timeframe('AAPL', ['1m', '15m', '1h', '1d'], '2024-01-01', '2024-12-01')
        >>>
        >>> # Compute velocity targets
        >>> targets = compute_velocity_targets(bars)
        >>> print(targets.head())
        >>>
        >>> # Use for training
        >>> # features = extract_features(bars)
        >>> # dataset = create_dataset(features, targets)
        >>> # model = train_finvec(dataset)
    """
    if timeframes is None:
        timeframes = ["1m", "15m", "1h", "1d"]

    # Timeframe specifications (max lookahead bars for each)
    timeframe_specs = {
        "1m": 30,  # Look ahead 30 bars (30 minutes)
        "15m": 16,  # Look ahead 16 bars (4 hours)
        "1h": 8,  # Look ahead 8 bars (1 trading day)
        "1d": 5,  # Look ahead 5 bars (1 trading week)
    }

    all_targets = []

    for tf in timeframes:
        if tf not in bars:
            logger.warning(f"Missing data for timeframe {tf}")
            continue

        df = bars[tf]

        if df is None or df.empty:
            logger.warning(f"Empty data for timeframe {tf}")
            continue

        max_lookahead = timeframe_specs.get(tf, 10)

        # Compute velocity targets for this timeframe
        targets = _compute_single_timeframe_targets(df, tf, max_lookahead)

        all_targets.append(targets)

    if not all_targets:
        logger.error("No targets computed for any timeframe")
        return pd.DataFrame()

    # Merge all targets
    # If different timeframes have different indices, we'll do an outer join
    # and forward-fill to align coarser timeframes to finer ones
    result = all_targets[0]

    for targets in all_targets[1:]:
        result = result.join(targets, how="outer")

    # Forward fill missing values (coarser timeframes will have NaNs for fine-grained timestamps)
    result = result.ffill()

    # Also backfill to handle start of series
    result = result.bfill()

    # Replace any remaining NaNs with zeros
    result = result.fillna(0)

    logger.info(
        f"Computed velocity targets: {result.shape[0]} rows, {result.shape[1]} columns"
    )

    return result


def _max_drawdown(path: np.ndarray) -> float:
    """Maximum drawdown along a price path (fraction 0..1)."""
    if len(path) < 2:
        return 0.0
    rm = np.maximum.accumulate(path)
    return float(abs(((path - rm) / (rm + 1e-8)).min()))


def _max_drawup(path: np.ndarray) -> float:
    """Maximum adverse excursion for a short position (fraction 0..1)."""
    if len(path) < 2:
        return 0.0
    rm = np.minimum.accumulate(path)
    return float(((path - rm) / (rm + 1e-8)).max())


def _compute_single_timeframe_targets(
    df: pd.DataFrame,
    timeframe_name: str,
    max_lookahead: int,
    price_col: str = "close",
    risk_penalty: float = 2.0,
    min_ret: float = 0.001,
) -> pd.DataFrame:
    """
    Compute RISK-ADJUSTED directional velocity targets for a single timeframe.

    For each bar the algorithm finds both the best LONG opportunity (peak with
    minimal drawdown on the path) and the best SHORT opportunity (valley with
    minimal drawup on the path) using the risk-adjusted score:

        score = |return| / (1 + risk_penalty * path_adverse_excursion)

    The WINNER — whichever direction has the higher score — becomes the
    dominant training target for that bar.  This ensures the model learns to
    predict both LONG and SHORT signals rather than always outputting LONG
    during periods when raw upside happens to be larger.

    Secondary-direction columns (long_velocity / short_velocity) are still
    populated for API compatibility and downstream reference, but the primary
    training signal is signed_velocity (positive = LONG, negative = SHORT).

    Args:
        df:             DataFrame with OHLCV data.
        timeframe_name: Prefix for output column names.
        max_lookahead:  Maximum bars to look ahead.
        price_col:      Price column (default 'close').
        risk_penalty:   Drawdown/drawup penalty multiplier (default 2.0).
        min_ret:        Minimum |return| for a bar to be a valid candidate.

    Returns:
        DataFrame with columns:
        - {tf}_signed_velocity : dominant signed velocity (+ LONG, − SHORT)
        - {tf}_signed_bars     : bars to dominant target bar
        - {tf}_direction       : +1.0 LONG / -1.0 SHORT
        - {tf}_confidence      : 1 / (1 + dominant path risk)
        - {tf}_long_velocity   : best long velocity ≥ 0  (API compat)
        - {tf}_long_bars       : bars to best long target
        - {tf}_short_velocity  : best short velocity ≤ 0 (API compat)
        - {tf}_short_bars      : bars to best short target
    """
    prices = df[price_col].values
    n = len(prices)

    signed_velocities = np.zeros(n)
    signed_bars_arr = np.ones(n, dtype=np.int32)
    directions = np.zeros(n)
    confidences = np.full(n, 0.5)
    long_velocities = np.zeros(n)
    long_bars_arr = np.ones(n, dtype=np.int32)
    short_velocities = np.zeros(n)
    short_bars_arr = np.ones(n, dtype=np.int32)

    for i in range(n - 1):
        lookahead = min(max_lookahead, n - i - 1)
        if lookahead < 1:
            continue

        cur = prices[i]
        if cur <= 0 or not np.isfinite(cur):
            continue

        future = prices[i + 1 : i + 1 + lookahead]
        rets = (future - cur) / (cur + 1e-8)

        # -------------------------------------------------- LONG
        best_ls, best_lj, best_lr, best_ldd = -np.inf, 0, 0.0, 0.0
        for j in range(len(rets)):
            r = rets[j]
            if r < min_ret:
                continue
            path = np.concatenate([[cur], future[: j + 1]])
            dd = _max_drawdown(path)
            s = r / (1.0 + risk_penalty * dd)
            if s > best_ls:
                best_ls, best_lj, best_lr, best_ldd = s, j, r, dd

        if best_ls == -np.inf:  # fallback: use highest raw return
            best_lj = int(np.argmax(rets))
            best_lr = max(0.0, float(rets[best_lj]))
            path = np.concatenate([[cur], future[: best_lj + 1]])
            best_ldd = _max_drawdown(path)
            best_ls = best_lr / (1.0 + risk_penalty * best_ldd) if best_lr > 0 else 0.0

        long_vel = best_lr / (best_lj + 1)
        long_conf = 1.0 / (1.0 + best_ldd)

        # ------------------------------------------------- SHORT
        best_ss, best_sj, best_sr, best_sdu = -np.inf, 0, 0.0, 0.0
        for j in range(len(rets)):
            r = rets[j]
            if r > -min_ret:
                continue
            path = np.concatenate([[cur], future[: j + 1]])
            du = _max_drawup(path)
            s = abs(r) / (1.0 + risk_penalty * du)
            if s > best_ss:
                best_ss, best_sj, best_sr, best_sdu = s, j, r, du

        if best_ss == -np.inf:  # fallback: use lowest raw return
            best_sj = int(np.argmin(rets))
            best_sr = min(0.0, float(rets[best_sj]))
            path = np.concatenate([[cur], future[: best_sj + 1]])
            best_sdu = _max_drawup(path)
            best_ss = (
                abs(best_sr) / (1.0 + risk_penalty * best_sdu) if best_sr < 0 else 0.0
            )

        short_vel = best_sr / (best_sj + 1)
        short_conf = 1.0 / (1.0 + best_sdu)

        # ----------------------------------------------- PICK WINNER
        if best_ls >= best_ss:
            dom_vel, dom_bars, dom_dir, dom_conf = long_vel, best_lj + 1, 1.0, long_conf
        else:
            dom_vel, dom_bars, dom_dir, dom_conf = (
                short_vel,
                best_sj + 1,
                -1.0,
                short_conf,
            )

        signed_velocities[i] = (
            np.clip(dom_vel, -2.0, 2.0) if np.isfinite(dom_vel) else 0.0
        )
        signed_bars_arr[i] = dom_bars
        directions[i] = dom_dir
        confidences[i] = dom_conf if np.isfinite(dom_conf) else 0.5
        long_velocities[i] = (
            np.clip(long_vel, 0.0, 2.0) if np.isfinite(long_vel) else 0.0
        )
        long_bars_arr[i] = best_lj + 1
        short_velocities[i] = (
            np.clip(short_vel, -2.0, 0.0) if np.isfinite(short_vel) else 0.0
        )
        short_bars_arr[i] = best_sj + 1

    p = timeframe_name
    return pd.DataFrame(
        {
            f"{p}_signed_velocity": signed_velocities,
            f"{p}_signed_bars": signed_bars_arr,
            f"{p}_direction": directions,
            f"{p}_confidence": confidences,
            f"{p}_long_velocity": long_velocities,
            f"{p}_long_bars": long_bars_arr,
            f"{p}_short_velocity": short_velocities,
            f"{p}_short_bars": short_bars_arr,
        },
        index=df.index,
    )


def validate_targets(targets: pd.DataFrame) -> Dict[str, any]:
    """
    Validate computed velocity targets for sanity.

    Checks for:
    - NaN values
    - Infinite values
    - Extreme values (outside expected range)
    - Empty data

    Args:
        targets: DataFrame with velocity targets

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'issues': [str],
            'stats': {
                'total_rows': int,
                'nan_count': int,
                'inf_count': int,
                'extreme_count': int,
            }
        }

    Example:
        >>> targets = compute_velocity_targets(bars)
        >>> validation = validate_targets(targets)
        >>> if not validation['valid']:
        >>>     print(f"Issues found: {validation['issues']}")
    """
    issues = []
    stats = {
        "total_rows": len(targets),
        "nan_count": 0,
        "inf_count": 0,
        "extreme_count": 0,
    }

    if targets.empty:
        issues.append("Targets DataFrame is empty")
        return {"valid": False, "issues": issues, "stats": stats}

    # Check for NaN values
    nan_count = targets.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values")
        stats["nan_count"] = nan_count

    # Check for infinite values
    inf_count = np.isinf(targets.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")
        stats["inf_count"] = inf_count

    # Check for extreme values (velocity > 200% or < -200%)
    velocity_cols = [col for col in targets.columns if "velocity" in col]
    extreme_count = 0

    for col in velocity_cols:
        extreme = ((targets[col] > 2.0) | (targets[col] < -2.0)).sum()
        extreme_count += extreme

    if extreme_count > 0:
        issues.append(f"Found {extreme_count} extreme velocity values (|v| > 2.0)")
        stats["extreme_count"] = extreme_count

    valid = len(issues) == 0

    if valid:
        logger.info("Target validation passed")
    else:
        logger.warning(f"Target validation issues: {issues}")

    return {
        "valid": valid,
        "issues": issues,
        "stats": stats,
    }


def compute_velocity_targets_per_symbol(
    bars_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """
    Compute velocity targets for each symbol.

    This is a wrapper for training orchestration that handles symbol-based input.

    Args:
        bars_by_symbol: Dict of symbol -> OHLCV DataFrame

    Returns:
        Dict of symbol -> velocity targets array [N, 8]
        (8 = 4 timeframes × 2 velocities per timeframe)
    """
    targets_by_symbol = {}

    for symbol, df in bars_by_symbol.items():
        # For single-timeframe data, compute forward returns at multiple horizons
        # This creates synthetic multi-timeframe targets
        try:
            # Compute simple velocity targets from forward returns
            close = df["close"].values
            n = len(close)

            # Horizons (bars to look ahead)
            horizons = [1, 5, 15, 60]  # ~1min, 5min, 15min, 1hour (assuming 1-min bars)

            velocities = []

            for horizon in horizons:
                # Long velocity (positive returns)
                long_vel = np.zeros(n)
                for i in range(n - horizon):
                    if close[i] != 0:  # Guard against division by zero
                        future_prices = close[i + 1 : i + horizon + 1]
                        max_return = (np.max(future_prices) / close[i]) - 1.0
                        long_vel[i] = max_return / horizon

                # Short velocity (negative returns - already negative when prices drop)
                short_vel = np.zeros(n)
                for i in range(n - horizon):
                    if close[i] != 0:  # Guard against division by zero
                        future_prices = close[i + 1 : i + horizon + 1]
                        min_return = (np.min(future_prices) / close[i]) - 1.0
                        short_vel[i] = (
                            min_return / horizon
                        )  # No negation - already negative for drops

                velocities.append(long_vel)
                velocities.append(short_vel)

            # Stack into array [N, 8]
            targets_array = np.column_stack(velocities)

            # Remove rows where we can't compute forward returns
            max_horizon = max(horizons)
            targets_array = targets_array[:-max_horizon]

            targets_by_symbol[symbol] = targets_array.astype(np.float32)

            logger.info(f"Computed targets for {symbol}: {targets_array.shape}")

        except Exception as e:
            logger.error(f"Failed to compute targets for {symbol}: {e}")
            continue

    return targets_by_symbol


def align_features_and_targets(
    features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align features and targets across all symbols.

    Handles:
    - Different lengths (due to forward return computation)
    - Missing symbols
    - NaN/Inf cleanup

    Args:
        features: Dict of symbol -> features [N, 361]
        targets: Dict of symbol -> targets [N, 8 or 10]

    Returns:
        (all_features, all_targets) stacked across symbols
    """
    all_features = []
    all_targets = []

    # Find common symbols
    common_symbols = set(features.keys()) & set(targets.keys())

    if not common_symbols:
        logger.error("No common symbols between features and targets!")
        return np.array([]), np.array([])

    logger.info(f"Aligning {len(common_symbols)} symbols")

    for symbol in sorted(common_symbols):
        feat = features[symbol]
        tgt = targets[symbol]

        # Align lengths (features may be longer due to forward computation)
        min_len = min(len(feat), len(tgt))

        if min_len == 0:
            logger.warning(f"Empty alignment for {symbol}, skipping")
            continue

        # Truncate to matching length
        feat_aligned = feat[:min_len]
        tgt_aligned = tgt[:min_len]

        # Remove NaN/Inf rows
        valid_mask = ~(
            np.isnan(feat_aligned).any(axis=1) | np.isnan(tgt_aligned).any(axis=1)
        )
        valid_mask &= ~(
            np.isinf(feat_aligned).any(axis=1) | np.isinf(tgt_aligned).any(axis=1)
        )

        feat_clean = feat_aligned[valid_mask]
        tgt_clean = tgt_aligned[valid_mask]

        if len(feat_clean) == 0:
            logger.warning(f"No valid samples for {symbol} after cleanup")
            continue

        all_features.append(feat_clean)
        all_targets.append(tgt_clean)

        logger.info(f"Aligned {symbol}: {len(feat_clean)} samples")

    if not all_features:
        logger.error("No valid features after alignment!")
        return np.array([]), np.array([])

    # Stack all symbols
    features_array = np.vstack(all_features)
    targets_array = np.vstack(all_targets)

    logger.info(
        f"Final dataset: features={features_array.shape}, targets={targets_array.shape}"
    )

    return features_array, targets_array
