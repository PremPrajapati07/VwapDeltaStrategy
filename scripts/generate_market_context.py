import os
import re
import datetime as dt
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

RAW_DATA_ROOT = "raw_data"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

def generate_context():
    print("🔍 Generating Market Context (Nifty Open) for Mar-Dec 2025 ...")
    
    daily_opens = {} # {date: {strikes: [list of synth spots at 9:15]}}
    
    # 1. Scan folders to find sample open prices
    for month in MONTHS:
        month_path = os.path.join(RAW_DATA_ROOT, month)
        if not os.path.exists(month_path):
            continue
            
        print(f"  Scanning {month} ...")
        files = [f for f in os.listdir(month_path) if f.endswith(".csv")]
        
        # We only need a few files per month to find all trade dates and their opens
        # Let's take CE/PE pairs for a likely ATM strike like 22000 or similar
        # To be safe, we'll scan all files but only process the first 9:15 AM candle we find for each date
        
        for f in tqdm(files, desc=f"Processing {month}"):
            m = re.search(r"NIFTY(\d{2})-(\d{2})-(\d{2})(\d+)(CE|PE)", f)
            if not m:
                m = re.search(r"NIFTY(\d{2})(\d{2})(\d{2})(\d+)(CE|PE)", f)
            if not m: continue
            
            strike = int(m.groups()[-2])
            otype = m.groups()[-1]
            
            # Load only enough to find 9:15 candles
            # Format: YYYYMMDD, HH:MM, O, H, L, C, V, OI
            try:
                # Optimized read: just date, time, and close
                df = pd.read_csv(os.path.join(month_path, f), header=None, usecols=[0, 1, 5], names=["date", "time", "close"])
                df_915 = df[df["time"] == "09:15"]
                
                for _, row in df_915.iterrows():
                    d_str = str(row["date"])
                    trade_date = dt.date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:]))
                    
                    if trade_date not in daily_opens:
                        daily_opens[trade_date] = {}
                    
                    if strike not in daily_opens[trade_date]:
                        daily_opens[trade_date][strike] = {}
                    
                    daily_opens[trade_date][strike][otype] = float(row["close"])
            except Exception:
                continue

    # 2. Derive Nifty Open from Synthetic Spot
    context_to_insert = []
    print("\n📊 Deriving Nifty Open for each day ...")
    for trade_date in sorted(daily_opens.keys()):
        # Find the strike that has both CE and PE at 9:15
        best_nifty_open = None
        
        # Try to find a strike near the median to avoid outliers
        for strike, legs in daily_opens[trade_date].items():
            if "CE" in legs and "PE" in legs:
                synth_spot = strike + legs["CE"] - legs["PE"]
                # We'll take the first stable one we find
                best_nifty_open = synth_spot
                break
        
        if best_nifty_open:
            context_to_insert.append((trade_date, round(best_nifty_open, 2), 15.0)) # VIX default 15
            print(f"  {trade_date}: {best_nifty_open:.2f}")

    # 3. Insert into DB
    if not context_to_insert:
        print("❌ No open prices found.")
        return

    print(f"\n🚀 Inserting {len(context_to_insert)} days into market_context ...")
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            execute_values(cur, """
                INSERT INTO market_context (trade_date, nifty_open, vix)
                VALUES %s
                ON CONFLICT (trade_date) DO UPDATE SET nifty_open = EXCLUDED.nifty_open
            """, context_to_insert)
    print("✅ Market context updated.")

if __name__ == "__main__":
    generate_context()
