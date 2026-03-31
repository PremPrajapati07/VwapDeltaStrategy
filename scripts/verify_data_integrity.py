import os
import sys
# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

def verify_integrity():
    print("🔍 VERIFYING DATA INTEGRITY (Jan-Feb 2025) ...")
    with db.get_conn() as conn:
        cur = conn.cursor()
        
        # 1. Total count for Jan
        cur.execute("SELECT COUNT(*) FROM straddle_candles WHERE trade_date BETWEEN '2025-01-01' AND '2025-01-31'")
        jan_total = cur.fetchone()[0]
        print(f"  Jan 2025 Total Rows: {jan_total:,}")
        
        # 2. Check for exact duplicates (same date, strike, minute)
        cur.execute("""
            SELECT trade_date, strike, ts, pg_column_size(ts), COUNT(*) 
            FROM straddle_candles 
            WHERE trade_date BETWEEN '2025-01-01' AND '2025-01-31'
            GROUP BY trade_date, strike, ts, pg_column_size(ts)
            HAVING COUNT(*) > 1
            LIMIT 5
        """)
        dupes = cur.fetchall()
        if dupes:
            print(f"  ❌ DUPLICATES CONFIRMED: Found up to {dupes[0][4]} copies for the same timestamp!")
            for d in dupes:
                print(f"    - {d[0]} | Strike {d[1]} | {d[2]} | {d[4]} copies")
        else:
            print("  ✅ No perfect duplicates found (at Postgres level).")

        # 3. Check for 'Close-but-different' timestamps (e.g. microseconds)
        cur.execute("""
            SELECT trade_date, strike, (EXTRACT(EPOCH FROM ts)::BIGINT / 60) as minute_epoch, COUNT(*) 
            FROM straddle_candles 
            WHERE trade_date BETWEEN '2025-01-01' AND '2025-01-31'
            GROUP BY trade_date, strike, minute_epoch
            HAVING COUNT(*) > 1
            LIMIT 5
        """)
        fuzzy_dupes = cur.fetchall()
        if fuzzy_dupes:
            print(f"  ⚠️ FUZZY DUPLICATES (Same Minute, different TS): Found {fuzzy_dupes[0][3]} rows in one minute!")

        # 4. Check unique constraint
        cur.execute("""
            SELECT conname, pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conrelid = 'straddle_candles'::regclass
        """)
        constraints = cur.fetchall()
        print("\n📜 Current DB Constraints:")
        for c in constraints:
            print(f"  - {c[0]}: {c[1]}")

if __name__ == "__main__":
    verify_integrity()
