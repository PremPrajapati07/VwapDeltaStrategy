import pandas as pd
import db
import krishna_model as ml
import datetime as dt

def debug_backtest():
    print("--- Debugging Backtest Data ---")
    feat_df = ml.build_features_from_db()
    print(f"feat_df columns: {feat_df.columns.tolist()}")
    if not feat_df.empty:
        print(f"feat_df['trade_date'] type: {type(feat_df.iloc[0]['trade_date'])}")
        print(f"feat_df['trade_date'] value: {feat_df.iloc[0]['trade_date']}")
    
    with db.get_conn() as conn:
        candles = pd.read_sql("SELECT * FROM straddle_candles LIMIT 10", conn)
    print(f"candles columns: {candles.columns.tolist()}")
    if not candles.empty:
        print(f"candles['trade_date'] type: {type(candles.iloc[0]['trade_date'])}")
        print(f"candles['trade_date'] value: {candles.iloc[0]['trade_date']}")
        
    if not feat_df.empty:
        print(f"Unique dates in feat_df: {feat_df['trade_date'].unique().tolist()}")
        # Check if 2026-03-25 is present
        mar25 = feat_df[feat_df["trade_date"].astype(str).str.contains("03-25")]
        print(f"March 25 rows: {len(mar25)}")

if __name__ == "__main__":
    debug_backtest()
