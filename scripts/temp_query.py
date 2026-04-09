import sys, os
sys.path.append(os.getcwd())
import db

with db.get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT nifty_prev_day_change, nifty_open_gap, vix FROM market_context WHERE trade_date = '2026-04-07'")
        row = cur.fetchone()
        print(f"nifty_prev_day_change: {row[0]}, nifty_open_gap: {row[1]}, vix: {row[2]}")
