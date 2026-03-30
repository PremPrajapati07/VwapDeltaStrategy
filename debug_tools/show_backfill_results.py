import db
import pandas as pd

def show_results():
    print("--- March 23-24 Backfill Results (Nifty Expiry Cycle) ---")
    with db.get_conn() as conn:
        df = pd.read_sql("""
            SELECT trade_date, strike, atm, straddle_premium, pnl, is_best
            FROM daily_features
            ORDER BY trade_date, strike
        """, conn)
    
    if df.empty:
        print("⚠️  No results found in daily_features.")
        return

    for date, grp in df.groupby("trade_date"):
        print(f"\n📅 Date: {date}")
        print(grp[["strike", "atm", "straddle_premium", "pnl", "is_best"]].to_string(index=False))
        
        best = grp[grp["is_best"]]
        if not best.empty:
            print(f"✅ Best Strike: {best.iloc[0]['strike']} (P&L: {best.iloc[0]['pnl']:.2f})")

if __name__ == "__main__":
    show_results()
