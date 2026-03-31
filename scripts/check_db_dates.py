import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2

# Local imports
import sys
sys.path.append('.')
import db

def check_dates():
    load_dotenv()
    print(f"Connecting to: {os.getenv('DATABASE_URL')}")
    try:
        with db.get_conn() as conn:
            df = pd.read_sql("SELECT DISTINCT trade_date FROM straddle_candles ORDER BY trade_date DESC LIMIT 10", conn)
            print("\nAvailable dates in straddle_candles:")
            print(df)
            
            # Also check market_context
            df_ctx = pd.read_sql("SELECT DISTINCT trade_date FROM market_context ORDER BY trade_date DESC LIMIT 5", conn)
            print("\nAvailable dates in market_context:")
            print(df_ctx)
    except Exception as e:
        print(f"Error checking dates: {e}")

if __name__ == "__main__":
    check_dates()
