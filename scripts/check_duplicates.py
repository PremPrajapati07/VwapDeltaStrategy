import os
import sys
# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import pandas as pd

def check():
    print("🔍 Checking for Duplicate Candles in Jan 2025 ...")
    with db.get_conn() as conn:
        # Count total rows
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM straddle_candles WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-01-31'")
        total = cur.fetchone()[0]
        print(f"  Total Jan rows: {total:,}")
        
        # Check for duplicates on a sample hour/strike
        cur.execute("""
            SELECT trade_date, strike, ts, COUNT(*) 
            FROM straddle_candles 
            WHERE trade_date = '2025-01-09' 
            GROUP BY trade_date, strike, ts 
            HAVING COUNT(*) > 1 
            LIMIT 5
        """)
        dupes = cur.fetchall()
        if dupes:
            print(f"  ❌ DUPLICATES FOUND! (Sample: {dupes[0][:3]} has {dupes[0][3]} copies)")
        else:
            print("  ✅ No duplicates found for Jan 9th.")

        # Check Schema
        cur.execute("""
            SELECT
                indexname,
                indexdef
            FROM
                pg_indexes
            WHERE
                tablename = 'straddle_candles';
        """)
        indexes = cur.fetchall()
        print("\n📜 Table Indexes:")
        for idx in indexes:
            print(f"  - {idx[0]}: {idx[1]}")

if __name__ == "__main__":
    check()
