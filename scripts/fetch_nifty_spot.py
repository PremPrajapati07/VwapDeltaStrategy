import sys
import os
import datetime as dt

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kite_auth
import db

def fetch_market_context():
    print("🚀 Fetching Nifty 50 and India VIX Daily data ...")
    
    kite = kite_auth.get_kite()
    nifty_token = 256265
    vix_token   = 264969
    
    # Range for backfill
    start_date = dt.date(2025, 1, 1)
    end_date = dt.date(2025, 2, 28)
    
    print(f"  Fetching from {start_date} to {end_date} ...")
    
    try:
        # 1. Fetch Nifty Daily Candles
        nifty_candles = kite.historical_data(nifty_token, start_date, end_date, "day")
        # 2. Fetch VIX Daily Candles
        vix_candles = kite.historical_data(vix_token, start_date, end_date, "day")
        
        if not nifty_candles:
            print("  ❌ No Nifty data received from Kite.")
            return

        # Map VIX by date for easy lookup
        vix_map = {c['date'].date(): float(c['close']) for c in vix_candles}

        print(f"  Processing {len(nifty_candles)} days ...")
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                for i, c in enumerate(nifty_candles):
                    trade_date = c['date'].date()
                    nifty_open = float(c['open'])
                    n_close    = float(c['close'])
                    
                    vix = vix_map.get(trade_date, 0.0)
                    
                    # Prev close is nifty_candles[i-1]['close']
                    # For the very first day, we might need to fetch one extra day
                    prev_close = 0.0
                    if i > 0:
                        prev_close = float(nifty_candles[i-1]['close'])
                    else:
                        # Try to fetch one candle before start_date if needed
                        extra_start = start_date - dt.timedelta(days=5)
                        extra = kite.historical_data(nifty_token, extra_start, start_date, "day")
                        if extra:
                            # Filter for last one before start_date
                            extra_c = [ec for ec in extra if ec['date'].date() < start_date]
                            if extra_c:
                                prev_close = float(extra_c[-1]['close'])
                        
                    print(f"    {trade_date}: Open={nifty_open:0.2f}, PrevClose={prev_close:0.2f}, VIX={vix:0.2f}")
                    
                    # Calculate gaps/change for ML consistency
                    prev_day_change = round((nifty_open - prev_close) / prev_close * 100, 4) if prev_close else 0.0
                    
                    cur.execute("""
                        INSERT INTO market_context (trade_date, nifty_open, nifty_prev_close, nifty_prev_day_change, vix)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (trade_date) DO UPDATE SET
                            nifty_open            = EXCLUDED.nifty_open,
                            nifty_prev_close      = EXCLUDED.nifty_prev_close,
                            nifty_prev_day_change = EXCLUDED.nifty_prev_day_change,
                            vix                   = EXCLUDED.vix
                    """, (trade_date, nifty_open, prev_close, prev_day_change, vix))
                    
        print("✅ Market context updated in database.")

    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    fetch_market_context()
