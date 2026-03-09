#!/usr/bin/env python3
"""
Intelligent Historical Data Collector

Smart symbol selection and automatic collection without manual symbol lists.
"""

import sys
import random
from datetime import datetime
from pathlib import Path
import yfinance as yf
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from fincoll.config.data_symbols import get_symbol_list


def get_random_symbols(source_list: str, count: int, exclude_symbols: list = None) -> list:
    """Get N random symbols from specified list."""
    exclude_symbols = exclude_symbols or ['AAPL', 'MSFT', 'GOOGL']
    
    all_symbols = get_symbol_list(source_list)
    available = [s for s in all_symbols if s not in exclude_symbols]
    
    if count > len(available):
        count = len(available)
    
    selected = random.sample(available, count)
    return selected


def get_market_cap_symbols(count: int, min_market_cap: float = 10e9) -> list:
    """Get symbols by market cap (top N)."""
    # Get top symbols by volume/market cap from yfinance
    common_mega_caps = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'
    ]
    
    if count <= len(common_mega_caps):
        return common_mega_caps[:count]
    
    # For more symbols, expand to large caps
    large_caps = [
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
        'CRM', 'NFLX', 'CMCSA', 'PEP', 'VZ', 'T', 'XOM', 'BAC', 'KO', 'PFE'
    ]
    
    all_symbols = common_mega_caps + large_caps
    return all_symbols[:count]


def get_sector_symbols(sectors: list, count_per_sector: int = 3) -> list:
    """Get symbols by sector."""
    sector_symbols = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM'],
        'financial': ['JPM', 'BAC', 'V', 'MA', 'WFC', 'GS', 'MS'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'BMY'],
        'consumer': ['AMZN', 'TSLA', 'HD', 'KO', 'PEP', 'PG', 'WMT'],
        'industrial': ['BA', 'CAT', 'GE', 'MMM', 'UNP'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB']
    }
    
    selected = []
    for sector in sectors:
        if sector in sector_symbols:
            selected.extend(sector_symbols[sector][:count_per_sector])
    
    return list(set(selected))  # Remove duplicates


def validate_symbols(symbols: list) -> tuple:
    """Validate symbols and return (valid, invalid) lists."""
    valid = []
    invalid = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                valid.append(symbol)
            else:
                invalid.append(symbol)
        except:
            invalid.append(symbol)
    
    return valid, invalid


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent historical data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Symbol selection modes:
  --auto N              : Auto-select N diversified symbols
  --mega-cap N          : Top N mega-cap stocks
  --sector tech,finance : Symbols from specific sectors
  --symbols AAPL,MSFT   : Manual symbol list (traditional)
  --random-from full N  : N random symbols from full universe
        """
    )
    
    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        '--auto', '-a',
        type=int,
        metavar='N',
        help='Auto-select N diversified symbols (recommended)'
    )
    
    symbol_group.add_argument(
        '--mega-cap',
        type=int,
        metavar='N',
        help='Top N mega-cap stocks by market cap'
    )
    
    symbol_group.add_argument(
        '--sector',
        type=str,
        metavar='SECTORS',
        help='Comma-separated sectors: tech,financial,healthcare,consumer,industrial,energy'
    )
    
    symbol_group.add_argument(
        '--random-from',
        nargs=2,
        metavar=('LIST', 'N'),
        help='N random symbols from LIST (diversified, tech, full)'
    )
    
    symbol_group.add_argument(
        '--symbols', '-s',
        nargs='+',
        help='Explicit symbol list (manual mode)'
    )
    
    # Other arguments
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
        '--validate',
        action='store_true',
        help='Validate symbols before downloading'
    )
    
    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help='Only estimate collection time'
    )
    
    args = parser.parse_args()
    
    # Determine symbols based on mode
    symbols = []
    selection_mode = "manual"
    
    if args.auto:
        symbols = get_random_symbols('diversified', args.auto)
        selection_mode = f"auto-{args.auto}"
    elif args.mega_cap:
        symbols = get_market_cap_symbols(args.mega_cap)
        selection_mode = f"mega-cap-{args.mega_cap}"
    elif args.sector:
        sectors = [s.strip() for s in args.sector.split(',')]
        symbols = get_sector_symbols(sectors)
        selection_mode = f"sectors-{len(sectors)}"
    elif args.random_from:
        source_list, count = args.random_from
        count = int(count)
        symbols = get_random_symbols(source_list, count)
        selection_mode = f"random-from-{source_list}-{count}"
    elif args.symbols:
        symbols = args.symbols
        selection_mode = f"manual-{len(symbols)}"
    else:
        # No arguments provided - use sensible default
        symbols = get_market_cap_symbols(10)
        selection_mode = "default-10-mega-caps"
    
    # Validate if requested
    if args.validate:
        print(f"\nValidating {len(symbols)} symbols...")
        valid, invalid = validate_symbols(symbols)
        
        if invalid:
            print(f"⚠️  {len(invalid)} invalid: {invalid}")
            print(f"✅ {len(valid)} valid: {valid}")
        else:
            print(f"✅ All {len(valid)} symbols validated successfully")
        
        symbols = valid
    
    # Show selected symbols
    print("\n" + "="*80)
    print(f"SYMBOL SELECTION: {selection_mode}")
    print("="*80)
    print(f"Count: {len(symbols)}")
    
    if len(symbols) <= 20:
        print(f"Symbols: {', '.join(symbols)}")
    else:
        print(f"Symbols: {', '.join(symbols[:15])}, ... ({len(symbols)-15} more)")
    print("="*80)
    
    # If --estimate-only with start/end dates provided, show full estimate
    if args.estimate_only or not any([args.start_date, args.end_date]):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from collect_historical_standalone import HistoricalDataCollector
        
        collector = HistoricalDataCollector(symbols, args.start_date, args.end_date)
        estimate = collector.estimate_collection_time()
        
        print("\nCOLLECTION ESTIMATE")
        print("="*80)
        print(f"Date range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
        print(f"Total samples: {estimate['total_samples']:,}")
        print(f"Estimated time: {estimate['estimated_time_hours']:.1f} hours")
        print("="*80)
        
        if args.estimate_only:
            return
    
    # Here you would normally call the actual collector
    print("\nSymbol selection complete. Use these symbols with collect_historical_standalone.py")
    print(f"\nExample:")
    symbols_arg = " ".join(symbols)
    print(f'python scripts/collect_historical_standalone.py --symbols {symbols_arg}')


if __name__ == "__main__":
    main()
