import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

def full_recovery_wipe():
    print("🧹 FULL 2025 DATABASE WIPE (RECOVERY MODE) ...")
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # 1. Clear Straddle Candles
                cur.execute("DELETE FROM straddle_candles WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")
                rows_candles = cur.rowcount
                
                # 2. Clear Market Context
                cur.execute("DELETE FROM market_context WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")
                rows_context = cur.rowcount
                
                # 3. Clear Daily Features (pre-calculated ML features)
                cur.execute("DELETE FROM daily_features WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")
                rows_features = cur.rowcount
                
                # 4. Clear Trade Log (historical trades)
                cur.execute("DELETE FROM trade_log WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")
                rows_trades = cur.rowcount

                conn.commit()
                print(f"✅ DB Wiped & Prepared for High-Fidelity Ingestion:")
                print(f"   - Removed {rows_candles:,} records from straddle_candles")
                print(f"   - Removed {rows_context:,} records from market_context")
                print(f"   - Removed {rows_features:,} records from daily_features")
                print(f"   - Removed {rows_trades:,} records from trade_log")
                
    except Exception as e:
        print(f"❌ Error during wipe: {e}")

if __name__ == "__main__":
    full_recovery_wipe()
