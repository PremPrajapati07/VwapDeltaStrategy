import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

def cleanup():
    print("🧹 Cleaning up March - December 2025 data from PostgreSQL...")
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Delete straddle_candles
                cur.execute("""
                    DELETE FROM straddle_candles 
                    WHERE trade_date >= '2025-03-01' 
                      AND trade_date <= '2025-12-31'
                """)
                rows_candles = cur.rowcount
                
                # Delete market_context
                cur.execute("""
                    DELETE FROM market_context 
                    WHERE trade_date >= '2025-03-01' 
                      AND trade_date <= '2025-12-31'
                """)
                rows_context = cur.rowcount
                
                conn.commit()
                print(f"✅ DB Cleaned!")
                print(f"   - Removed {rows_candles} records from straddle_candles")
                print(f"   - Removed {rows_context} records from market_context")
                
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    cleanup()
