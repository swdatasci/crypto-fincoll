#!/usr/bin/env python3
"""
Bulk Cache Pre-Warming Script for FinColl
==========================================

Fetches historical data for all symbols in parallel batches and stores in InfluxDB cache.
Run this before market open (8:00 AM ET) to ensure 100% cache hit rate during trading.

Usage:
    python prewarm_cache_bulk.py                    # Default: All tracked symbols
    python prewarm_cache_bulk.py --symbols AAPL GOOGL MSFT  # Specific symbols
    python prewarm_cache_bulk.py --batch-size 20    # Parallel batch size
"""

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add fincoll to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from typing import List, Dict, Any


class CachePreWarmer:
    """Pre-warm FinColl cache by fetching predictions for all symbols"""
    
    def __init__(self, fincoll_url: str = "http://10.32.3.27:8002", batch_size: int = 10):
        self.fincoll_url = fincoll_url
        self.batch_size = batch_size
        self.results = []
        
    async def fetch_prediction(self, client: httpx.AsyncClient, symbol: str) -> Dict[str, Any]:
        """Fetch prediction for a single symbol"""
        start = datetime.now()
        try:
            url = f"{self.fincoll_url}/api/v1/inference/predict/{symbol}"
            response = await client.post(url, timeout=60.0)
            response.raise_for_status()
            
            elapsed = (datetime.now() - start).total_seconds()
            print(f"✅ {symbol:6s} - {elapsed:5.1f}s")
            
            return {
                "symbol": symbol,
                "success": True,
                "elapsed": elapsed,
                "status": response.status_code
            }
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"❌ {symbol:6s} - {elapsed:5.1f}s - {str(e)[:50]}")
            
            return {
                "symbol": symbol,
                "success": False,
                "elapsed": elapsed,
                "error": str(e)
            }
    
    async def prewarm_batch(self, symbols: List[str]):
        """Pre-warm cache for a batch of symbols in parallel"""
        async with httpx.AsyncClient() as client:
            tasks = [self.fetch_prediction(client, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            self.results.extend(results)
    
    async def prewarm_all(self, symbols: List[str]):
        """Pre-warm cache for all symbols in batches"""
        print(f"\n🚀 Pre-warming cache for {len(symbols)} symbols")
        print(f"   Batch size: {self.batch_size}")
        print(f"   FinColl URL: {self.fincoll_url}\n")
        
        total_start = datetime.now()
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size
            
            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} symbols)")
            await self.prewarm_batch(batch)
        
        total_elapsed = (datetime.now() - total_start).total_seconds()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Pre-warming Summary")
        print("="*60)
        
        successes = sum(1 for r in self.results if r['success'])
        failures = len(self.results) - successes
        avg_time = sum(r['elapsed'] for r in self.results) / len(self.results)
        
        print(f"Total symbols:     {len(symbols)}")
        print(f"Successful:        {successes} ({successes/len(symbols)*100:.1f}%)")
        print(f"Failed:            {failures}")
        print(f"Total time:        {total_elapsed:.1f}s")
        print(f"Avg per symbol:    {avg_time:.1f}s")
        print(f"Effective rate:    {len(symbols)/total_elapsed:.2f} symbols/sec")
        
        if failures > 0:
            print("\n❌ Failed symbols:")
            for r in self.results:
                if not r['success']:
                    print(f"   {r['symbol']:6s} - {r.get('error', 'Unknown error')[:60]}")
        
        print("="*60 + "\n")


async def get_tracked_symbols_from_db() -> List[str]:
    """Fetch tracked symbols from PIM database"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host="10.32.3.27",
            port=15433,
            database="pim_database",
            user="pim_user",
            password="pim_password_2024"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM symbols_universe WHERE tracked = true ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return symbols
    except Exception as e:
        print(f"⚠️  Could not fetch from database: {e}")
        print("   Using default symbol list")
        return [
            'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'SPY', 'QQQ',
            'META', 'TSLA', 'NFLX', 'AMD', 'INTC', 'CSCO', 'AVGO'
        ]


def main():
    parser = argparse.ArgumentParser(description="Pre-warm FinColl cache for bulk symbols")
    parser.add_argument('--symbols', nargs='*', help="Specific symbols to pre-warm")
    parser.add_argument('--batch-size', type=int, default=10, help="Parallel batch size (default: 10)")
    parser.add_argument('--fincoll-url', default="http://10.32.3.27:8002", help="FinColl API URL")
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"📋 Using provided symbols: {symbols}")
    else:
        print("📋 Fetching tracked symbols from database...")
        symbols = asyncio.run(get_tracked_symbols_from_db())
    
    # Pre-warm cache
    prewarmer = CachePreWarmer(fincoll_url=args.fincoll_url, batch_size=args.batch_size)
    asyncio.run(prewarmer.prewarm_all(symbols))


if __name__ == "__main__":
    main()
