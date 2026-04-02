import sys
import os

# Add parent directory to sys.path to import db
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_conn

def clear_today(date_to_clear="2026-04-01"):
    tables = [
        "straddle_candles",
        "market_context",
        "daily_features",
        "trade_log",
        "krishna_predictions"
    ]
    
    print(f"🧹 Clearing all data for {date_to_clear} from database...")
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            for table in tables:
                # Some tables might use 'trade_date' exactly
                cur.execute(f"DELETE FROM {table} WHERE trade_date = %s", (date_to_clear,))
                print(f"   - Cleared {cur.rowcount} rows from {table}")
    
    print("✅ Today's data successfully wiped.")

if __name__ == "__main__":
    # If a date is passed as an argument, use it; otherwise use today's target date
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2026-04-01"
    clear_today(target_date)
