import db
import pandas as pd

def check_db():
    with db.get_conn() as conn:
        df_candles = pd.read_sql("SELECT trade_date, count(*) FROM straddle_candles GROUP BY trade_date", conn)
        df_feat = pd.read_sql("SELECT trade_date, count(*) FROM daily_features GROUP BY trade_date", conn)
        print("--- Straddle Candles ---")
        print(df_candles)
        print("\n--- Daily Features ---")
        print(df_feat)

if __name__ == "__main__":
    check_db()
