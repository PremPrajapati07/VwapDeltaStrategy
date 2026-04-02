import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_conn

with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT strike, COUNT(*) as rows,
                   ROUND(AVG(ce_open)::numeric, 2) as avg_ce_open,
                   ROUND(AVG(ce_iv)::numeric, 4) as avg_ce_iv,
                   ROUND(AVG(ce_delta)::numeric, 4) as avg_ce_delta,
                   ROUND(AVG(vwap)::numeric, 2) as avg_vwap
            FROM straddle_candles
            WHERE trade_date = '2026-04-01'
            GROUP BY strike ORDER BY strike
        """)
        rows = cur.fetchall()
        print(f"{'Strike':>10} {'Rows':>6} {'CE Open':>10} {'CE IV':>8} {'CE Delta':>10} {'VWAP':>10}")
        print("-" * 60)
        for r in rows:
            print(f"{r[0]:>10} {r[1]:>6} {str(r[2]):>10} {str(r[3]):>8} {str(r[4]):>10} {str(r[5]):>10}")
        print(f"\nTotal rows: {sum(r[1] for r in rows)}")
