#!/usr/bin/env python3
"""
Standalone Historical Data Collector (No Torch Dependencies)

Collects OHLCV data from yfinance directly, bypassing fincoll's torch-dependent
inference modules. Stores raw data that can later be processed into features.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import time
import logging
from tqdm import tqdm
import yfinance as yf

# Simple logger setup (no dependencies)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("historical_collector")


class SimpleInfluxDBClient:
    """Minimal InfluxDB client for storing raw OHLCV data."""
    
    def __init__(self, url: str = "http://10.32.3.27:8086", token: str = "", org: str = "caelum"):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = "training_data"
        
        # Try to import real client, fallback to mock
        try:
            from influxdb_client import InfluxDBClient, Point
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api()
            self.Point = Point
            self.available = True
        except ImportError:
            logger.warning("influxdb-client not available, using mock storage")
            self.available = False
            self.mock_storage = []
    
    def save_ohlcv(self, symbol: str, date: datetime, ohlcv: Dict[str, float], tags: Dict = None):
        """Save OHLCV data point to InfluxDB."""
        if not self.available:
            # Mock storage for testing
            self.mock_storage.append({
                'symbol': symbol,
                'date': date,
                'ohlcv': ohlcv,
                'tags': tags or {}
            })
            return True
        
        try:
            point = self.Point("ohlcv") \
                .tag("symbol", symbol) \
                .field("open", ohlcv['open']) \
                .field("high", ohlcv['high']) \
                .field("low", ohlcv['low']) \
                .field("close", ohlcv['close']) \
                .field("volume", ohlcv['volume']) \
                .time(date)
            
            # Add additional tags
            if tags:
                for tag_key, tag_value in tags.items():
                    point = point.tag(tag_key, tag_value)
            
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            logger.error(f"Failed to save OHLCV for {symbol}: {e}")
            return False
    
    def get_counts(self) -> Dict:
        """Get storage statistics."""
        if not self.available:
            symbols = set()
            for item in self.mock_storage:
                symbols.add(item['symbol'])
            
            return {
                'total_records': len(self.mock_storage),
                'unique_symbols': len(symbols),
                'storage_type': 'MOCK (influxdb-client not installed)'
            }
        
        return {
            'total_records': 'N/A (InfluxDB)',
            'unique_symbols': 'N/A (InfluxDB)',
            'storage_type': 'InfluxDB'
        }


class HistoricalDataCollector:
    """Collects historical OHLCV data from yfinance."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 10,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        
        # InfluxDB client
        self.db = SimpleInfluxDBClient()
        
        # Statistics
        self.stats = {
            "processed_symbols": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_days": 0,
            "start_time": datetime.now()
        }
    
    def collect_all(self, resume_from: Optional[datetime] = None):
        """Collect data for all symbols."""
        effective_start = resume_from or self.start_date
        
        logger.info("="*80)
        logger.info("HISTORICAL DATA COLLECTION (Standalone)")
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info(f"Date range: {effective_start.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info("="*80)
        
        # Process in batches
        for i in tqdm(range(0, len(self.symbols), self.batch_size), desc="Symbol batches"):
            batch_symbols = self.symbols[i:i + self.batch_size]
            logger.info(f"\nProcessing batch {i//self.batch_size + 1}: {len(batch_symbols)} symbols")
            
            for symbol in batch_symbols:
                self._process_symbol(symbol)
                time.sleep(0.5)  # Rate limit
                
        self._print_summary()
    
    def _process_symbol(self, symbol: str):
        """Download and store data for a single symbol."""
        logger.info(f"Processing {symbol}...")
        
        try:
            # Download from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get full historical data
            hist_data = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist_data.empty:
                logger.warning(f"No data for {symbol}")
                self.stats["failed_downloads"] += 1
                return
            
            # Save each row
            success_count = 0
            for date_idx in range(len(hist_data)):
                row = hist_data.iloc[date_idx]
                date = hist_data.index[date_idx].to_pydatetime()
                
                # Extract OHLCV
                ohlcv = {
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                }
                
                # Store
                if self.db.save_ohlcv(symbol, date, ohlcv):
                    success_count += 1
                    self.stats["total_days"] += 1
            
            self.stats["successful_downloads"] += 1
            logger.info(f"  → {success_count} days of data")
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            self.stats["failed_downloads"] += 1
    
    def estimate_collection_time(self) -> dict:
        """Estimate collection duration."""
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
    
    def _print_summary(self):
        """Print collection statistics."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
        db_stats = self.db.get_counts()
        
        print("\n" + "="*80)
        print("COLLECTION COMPLETE")
        print("="*80)
        print(f"Duration: {duration:.2f} hours")
        print(f"Symbols processed: {self.stats['processed_symbols']}/{len(self.symbols)}")
        print(f"Successful downloads: {self.stats['successful_downloads']}")
        print(f"Failed downloads: {self.stats['failed_downloads']}")
        print(f"Total days: {self.stats['total_days']:,}")
        print(f"Database: {db_stats['storage_type']}")
        if 'total_records' in db_stats:
            print(f"Total records: {db_stats['total_records']:,}")
            print(f"Unique symbols: {db_stats['unique_symbols']}")
        print("="*80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Collect historical OHLCV data (standalone, no torch deps)"
    )
    
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        help='Symbols to collect'
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
        help='End date (YYYY-MM-DD or "today")'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Symbols per batch (default: 10)'
    )
    
    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help='Only estimate collection time'
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = HistoricalDataCollector(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size
    )
    
    # Estimate
    estimate = collector.estimate_collection_time()
    print("\n" + "="*80)
    print("HISTORICAL DATA COLLECTION ESTIMATE")
    print("="*80)
    print(f"Symbols: {estimate['symbols']}")
    print(f"Days: {estimate['days']:,}")
    print(f"Total samples: {estimate['total_samples']:,}")
    print(f"Estimated time: {estimate['estimated_time_hours']:.1f} hours")
    print("="*80)
    
    # Check if collection is reasonable before market opens
    hours_until_market = 5.5  # Approx until 9:30 AM EST
    if estimate['estimated_time_hours'] > hours_until_market:
        print(f"\n⚠️  WARNING: Collection will take {estimate['estimated_time_hours']:.1f} hours")
        print(f"   Market opens in {hours_until_market} hours")
        print("   Consider collecting a subset of symbols or date range")
    
    if args.estimate_only:
        print("\nEstimate complete. Run without --estimate-only to collect data.")
        return
    
    # Confirm
    response = input("\nStart collection? (yes/no): ")
    if response.lower() != 'yes':
        print("Collection cancelled.")
        return
    
    # Start
    collector.collect_all()


if __name__ == "__main__":
    main()
