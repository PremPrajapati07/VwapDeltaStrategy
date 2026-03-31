import os
from dotenv import load_dotenv
import pandas as pd
import sys

# Local imports
sys.path.append('.')
import db

def verify():
    load_dotenv()
    date = "2026-03-30"
    try:
        with db.get_conn() as conn:
            c_count = pd.read_sql(f"SELECT COUNT(*) FROM straddle_candles WHERE trade_date = '{date}'", conn).iloc[0,0]
            f_count = pd.read_sql(f"SELECT COUNT(*) FROM daily_features WHERE trade_date = '{date}'", conn).iloc[0,0]
            ctx_count = pd.read_sql(f"SELECT COUNT(*) FROM market_context WHERE trade_date = '{date}'", conn).iloc[0,0]
            
            print(f"\n📊 March 30 Data Audit:")
            print(f"   straddle_candles : {c_count:,} rows")
            print(f"   daily_features   : {f_count:,} rows")
            print(f"   market_context   : {ctx_count:,} rows")
            
            if f_count == 0 and c_count > 0:
                print("\n⚠️  daily_features is EMPTY. I need to rebuild features for this date.")
            elif c_count == 0:
                print("\n❌ straddle_candles is EMPTY. Need to backfill.")
            else:
                print("\n✅ Data looks sufficient for simulation.")
                
    except Exception as e:
        print(f"Error verifying data: {e}")

if __name__ == "__main__":
    verify()
