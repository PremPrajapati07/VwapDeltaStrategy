import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import pandas as pd

def check_counts():
    with db.get_conn() as conn:
        print("📊 Global Database Status:")
        
        # 1. Check Candles
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), count(DISTINCT trade_date), count(DISTINCT strike) FROM straddle_candles")
            cand_count, dates, strikes = cur.fetchone()
            print(f"  Straddle Candles: {cand_count:,} rows across {dates} dates and {strikes} strikes.")
            
        # 2. Check Context
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM market_context")
            context_count = cur.fetchone()[0]
            print(f"  Market Context:   {context_count} rows total.")
            if context_count > 0:
                cur.execute("SELECT trade_date, nifty_open FROM market_context ORDER BY trade_date LIMIT 3")
                print(f"    Earliest Context: {[f'{str(d[0])}: {d[1]}' for d in cur.fetchall()]}")
                cur.execute("SELECT trade_date, nifty_open FROM market_context ORDER BY trade_date DESC LIMIT 3")
                print(f"    Latest Context:   {[f'{str(d[0])}: {d[1]}' for d in cur.fetchall()]}")

        # 3. Check Features
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM daily_features")
            feat_count = cur.fetchone()[0]
            print(f"  Daily Features:   {feat_count} rows.")

if __name__ == "__main__":
    check_counts()
