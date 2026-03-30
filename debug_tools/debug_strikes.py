import kite_auth
import data_collector as dc
import config
import pandas as pd
import datetime as dt

def debug_strikes():
    print("--- Debugging Expiry List ---")
    kite = kite_auth.get_kite()
    
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    
    nifty = df[df["name"] == "NIFTY"]
    print(f"DEBUG: Total NIFTY count: {len(nifty)}")
    
    # Filter for options specifically
    nifty_opts = nifty[nifty["instrument_type"].isin(["CE", "PE"])]
    expiries = sorted(nifty_opts["expiry"].dropna().unique())
    
    print("\nDEBUG: Active NIFTY Expiries (first 10):")
    for exp in expiries[:10]:
        # Print date and type of object
        print(f"   {exp} (Type: {type(exp)})")
        
    today = dt.date.today()
    print(f"\nDEBUG: Today's date: {today}")
    
    # Find the nearest expiry that is today or later
    nearest = [e for e in expiries if e >= today]
    if nearest:
        print(f"DEBUG: Nearest available expiry: {nearest[0]}")
    else:
        print("⚠️  No future expiries found for NIFTY!")

if __name__ == "__main__":
    debug_strikes()
