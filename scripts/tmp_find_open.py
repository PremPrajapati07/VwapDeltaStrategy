import os
import re
import pandas as pd
import datetime as dt

RAW_DIR = "raw_data/march"

def find_nifty_open():
    print("🔍 Searching for Nifty Open on 2025-03-04 (All Files)...")
    target_date = "20250304"
    target_time = "09:15"
    
    # Get all Files
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    
    # Groups by strike
    strikes = {}
    for f in files:
        m = re.search(r"NIFTY25-03-06(\d+)(CE|PE)", f)
        if m:
            strike, otype = m.groups()
            strike = int(strike)
            if strike not in strikes: strikes[strike] = {}
            strikes[strike][otype] = f

    for strike, legs in strikes.items():
        if "CE" in legs and "PE" in legs:
            ce_path = os.path.join(RAW_DIR, legs["CE"])
            pe_path = os.path.join(RAW_DIR, legs["PE"])
            
            ce = pd.read_csv(ce_path, header=None, names=["d","t","o","h","l","c","v","oi"])
            pe = pd.read_csv(pe_path, header=None, names=["d","t","o","h","l","c","v","oi"])
            
            ce_915 = ce[(ce["d"].astype(str) == target_date) & (ce["t"] == target_time)]
            pe_915 = pe[(pe["d"].astype(str) == target_date) & (pe["t"] == target_time)]
            
            if not ce_915.empty and not pe_915.empty:
                spot = strike + ce_915.iloc[0]["c"] - pe_915.iloc[0]["c"]
                print(f"✅ Found Nifty Open on strike {strike}: {spot:.2f}")
                return spot
                
    print("❌ Could not find 9:15 data in any file.")
    return None

if __name__ == "__main__":
    find_nifty_open()
