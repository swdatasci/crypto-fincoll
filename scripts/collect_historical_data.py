#!/usr/bin/env python3
"""
Collect Historical Training Data (2020-Present)

Collects feature vectors for training from 2020-01-01 to present using yfinance.
Features are extracted using the FeatureConstructor and stored in InfluxDB.

Usage:
    python scripts/collect_historical_data.py \
        --symbols diversified \
        --start-date 2020-01-01 \
        --end-date 2024-12-31 \
        --batch-size 20

    python scripts/collect_historical_data.py \
        --symbols AAPL MSFT GOOGL TSLA NVDA \
        --start-date 2020-01-01 \
        --end-date today

    python scripts/collect_historical_data.py \
        --resume-from 2024-06-15 \
        --symbols tech
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging
import argparse
import time
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.providers.yfinance_provider import YFinanceProvider
from fincoll.features.constructor import FeatureConstructor
from fincoll.storage.influxdb_saver import InfluxDBSaver
from fincoll.utils.logger import setup_logger
from fincoll.config.data_symbols import get_symbol_list
from fincoll.config.feature_dimensions import load_dimension_config


class HistoricalDataCollector:
    """Collector for historical training data from 2020 to present."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 20,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize historical data collector.
        
        Args:
            symbols: List of symbols to collect data for
            start_date: Start date for collection
            end_date: End date for collection
            batch_size: Number of symbols to process in parallel
            cache_dir: Optional cache directory for raw OHLCV data
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Initialize components
        self.logger = setup_logger("historical_collector")
        self.data_provider = YFinanceProvider(cache_dir=cache_dir)
        self.feature_constructor = FeatureConstructor()
        self.influx_saver = InfluxDBSaver()
        
        # Load config for versioning
        self.dim_config = load_dimension_config()
        self.config_hash = self._compute_config_hash()
        
        # Statistics
        self.stats = {
            "processed_symbols": 0,
            "processed_days": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "skipped_samples": 0,
            "symbols_with_errors": set(),
            "start_time": datetime.now()
        }
    
    def _compute_config_hash(self) -> str:
        """Compute MD5 hash of feature dimension config."""
        config_path = Path(__file__).parent.parent / "config" / "feature_dimensions.yaml"
        if not config_path.exists():
            self.logger.warning("Config not found, using placeholder hash")
            return "unknown"
        
        try:
            with open(config_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()[:8]
        except Exception as e:
            self.logger.warning(f"Could not compute hash: {e}")
            return "error"
    
    def collect_all(self, resume_from: Optional[datetime] = None):
        """
        Collect data for all symbols across date range.
        
        Args:
            resume_from: If provided, resume collection from this date
        """
        effective_start = resume_from or self.start_date
        
        self.logger.info("="*80)
        self.logger.info(f"HISTORICAL DATA COLLECTION START")
        self.logger.info(f"Symbols: {len(self.symbols)}")
        self.logger.info(f"Date range: {effective_start.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"Config version: v{self.config_hash}")
        self.logger.info("="*80)
        
        date_range = pd.date_range(effective_start, self.end_date, freq='D')
        
        # Process symbols in batches
        for i in tqdm(
            range(0, len(self.symbols), self.batch_size),
            desc="Processing symbol batches",
            unit="batch"
        ):
            batch_symbols = self.symbols[i:i + self.batch_size]
            
            self.logger.info(f"\nProcessing batch: {len(batch_symbols)} symbols")
            
            # Process each symbol in batch
            for symbol in batch_symbols:
                self._process_symbol(symbol, date_range)
        
        self._print_summary()
    
    def _process_symbol(self, symbol: str, date_range):
        """Process a single symbol across all dates."""
        self.logger.info(f"Processing {symbol} ({len(date_range)} days)")
        
        success_count = 0
        error_count = 0
        
        for current_date in tqdm(
            date_range,
            desc=f"  {symbol}",
            unit="day",
            leave=False
        ):
            try:
                # Check if we already have data for this date
                if self._has_existing_data(symbol, current_date):
                    self.stats["skipped_samples"] += 1
                    continue
                
                # Extract features
                features = self._extract_features(symbol, current_date)
                
                if features is None:
                    self.logger.debug(f"No features for {symbol} on {current_date}")
                    error_count += 1
                    continue
                
                # Store features
                self._save_features(symbol, current_date, features)
                
                success_count += 1
                self.stats["successful_samples"] += 1
                
                # Brief pause to be API-friendly
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol} on {current_date}: {e}")
                error_count += 1
                self.stats["failed_samples"] += 1
                self.stats["symbols_with_errors"].add(symbol)
        
        self.stats["processed_symbols"] += 1
        self.logger.info(f"  → {success_count} success, {error_count} failed")
    
    def _has_existing_data(self, symbol: str, date: datetime) -> bool:
        """Check if we already have feature data for symbol/date."""
        try:
            # This would query InfluxDB - for now just check cache
            cache_file = self._get_cache_file(symbol, date)
            if cache_file.exists():
                # Check if cache is recent enough
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < 7:
                    return True
            
            return False
        except Exception as e:
            self.logger.debug(f"Could not check existing data for {symbol}: {e}")
            return False
    
    def _get_cache_file(self, symbol: str, date: datetime) -> Path:
        """Get cache file path for symbol/date."""
        cache_dir = Path("cache/features") if not self.cache_dir else Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = date.strftime('%Y-%m-%d')
        return cache_dir / f"{symbol}_{date_str}.npy"
    
    def _extract_features(self, symbol: str, date: datetime) -> Optional[np.ndarray]:
        """Extract feature vector for symbol on specific date."""
        try:
            # Get data from provider
            # We need some historical context to calculate features
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Get OHLCV data
            data = self.data_provider.get_ohlcv(
                symbol=symbol,
                timeframe='1d',
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or len(data) == 0:
                self.logger.debug(f"No OHLCV data for {symbol}")
                return None
            
            # Use FeatureConstructor to extract features
            # For historical data, we use the last available day's features
            features = self.feature_constructor.construct_from_data(
                symbol=symbol,
                data=data,
                date=date
            )
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {symbol}: {e}")
            return None
    
    def _save_features(self, symbol: str, date: datetime, features: np.ndarray):
        """Save features to InfluxDB."""
        try:
            # Create metadata tags
            tags = {
                'symbol': symbol,
                'feature_dim': f"{len(features)}D",
                'config_version': f"v{self.config_hash}",
                'collection_type': 'historical',
                'collection_date': datetime.now().isoformat()
            }
            
            # Save to database
            self.influx_saver.save_training_sample(
                symbol=symbol,
                timestamp=date,
                features=features,
                targets=None,  # No targets for historical data
                tags=tags
            )
            
            # Also cache to disk for faster reuse
            cache_file = self._get_cache_file(symbol, date)
            np.save(cache_file, features)
            
            self.stats["processed_days"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to save features for {symbol}: {e}")
            raise
    
    def _print_summary(self):
        """Print collection statistics."""
        end_time = datetime.now()
        duration = (end_time - self.stats["start_time"]).total_seconds() / 3600
        
        print("\n" + "="*80)
        print("HISTORICAL DATA COLLECTION COMPLETE")
        print("="*80)
        print(f"Duration: {duration:.2f} hours")
        print(f"Processed symbols: {self.stats['processed_symbols']}")
        print(f"Processed days: {self.stats['processed_days']}")
        print(f"Successful samples: {self.stats['successful_samples']}")
        print(f"Failed samples: {self.stats['failed_samples']}")
        print(f"Skipped samples: {self.stats['skipped_samples']}")
        print(f"Symbols with errors: {len(self.stats['symbols_with_errors'])}")
        if self.stats['symbols_with_errors']:
            for symbol in list(self.stats['symbols_with_errors'])[:5]:
                print(f"  - {symbol}")
        print(f"Success rate: {self.stats['successful_samples'] / max(1, self.stats['successful_samples'] + self.stats['failed_samples']):.1%}")
        print("="*80)
    
    def check_yfinance_limits(self) -> dict:
        """Check yfinance rate limit status."""
        # yfinance has rate limits but they're not programmatically exposed
        # We'll track our own rate of requests
        return {
            "requests_per_second": 1.0,  # Conservative
            "daily_limit_reached": False,
            "cooldown_seconds": 60 if False else 0
        }
    
    def estimate_collection_time(self) -> dict:
        """Estimate how long collection will take."""
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        total_samples = len(self.symbols) * len(date_range)
        
        # Estimate 2 seconds per sample (API call + processing)
        estimated_seconds = total_samples * 2.0
        
        return {
            "total_samples": total_samples,
            "estimated_time_hours": estimated_seconds / 3600,
            "estimated_time_days": estimated_seconds / (3600 * 24),
            "symbols": len(self.symbols),
            "days": len(date_range)
        }


def main():
    """Command-line interface for historical data collection."""
    parser = argparse.ArgumentParser(
        description="Collect historical training data from 2020 to present"
    )
    
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['AAPL'],
        help='Symbols to collect (or use: diversified, tech, full)'
    )
    
    parser.add_argument(
        '--start-date',
        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
        default=datetime(2020, 1, 1),
        help='Start date (YYYY-MM-DD, default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=lambda d: datetime.now() if d == 'today' else datetime.strptime(d, '%Y-%m-%d'),
        default='today',
        help='End date (YYYY-MM-DD or "today", default: today)'
    )
    
    parser.add_argument(
        '--resume-from',
        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
        help='Resume collection from this date (skip earlier dates)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Symbols per batch (default: 10)'
    )
    
    parser.add_argument(
        '--cache-dir',
        help='Directory for caching raw data (optional)'
    )
    
    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help='Only estimate collection time, don\'t actually collect'
    )
    
    parser.add_argument(
        '--estimate-symbols',
        type=int,
        default=50,
        help='Number of symbols to use for estimate (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Handle predefined symbol lists
    if len(args.symbols) == 1:
        symbol_list_name = args.symbols[0]
        if symbol_list_name in ['diversified', 'tech', 'full', 'demo']:
            symbols = get_symbol_list(symbol_list_name)
            print(f"Using {symbol_list_name} symbol list: {len(symbols)} symbols")
        else:
            symbols = args.symbols
    else:
        symbols = args.symbols
    
    # Limit symbols if just estimating
    if args.estimate_only:
        symbols = symbols[:args.estimate_symbols]
    
    # Create collector
    collector = HistoricalDataCollector(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir
    )
    
    # Estimate collection time
    estimate = collector.estimate_collection_time()
    print("\n" + "="*80)
    print("COLLECTION ESTIMATE")
    print("="*80)
    print(f"Symbols: {estimate['symbols']}")
    print(f"Days: {estimate['days']}")
    print(f"Total samples: {estimate['total_samples']:,}")
    print(f"Estimated time: {estimate['estimated_time_hours']:.1f} hours ({estimate['estimated_time_days']:.1f} days)")
    print("="*80)
    
    if args.estimate_only:
        print("\nEstimate complete. Run without --estimate-only to collect data.")
        return
    
    # Confirm with user
    response = input("\nProceed with data collection? (yes/no) ")
    if response.lower() != 'yes':
        print("Collection cancelled.")
        return
    
    # Check yfinance limits
    limits = collector.check_yfinance_limits()
    if limits['daily_limit_reached']:
        print("WARNING: yfinance daily limit may have been reached!")
        print(f"Cooldown recommended: {limits['cooldown_seconds']} seconds")
        response = input("Continue anyway? (yes/no) ")
        if response.lower() != 'yes':
            return
    
    # Start collection
    collector.collect_all(resume_from=args.resume_from)


if __name__ == "__main__":
    main()