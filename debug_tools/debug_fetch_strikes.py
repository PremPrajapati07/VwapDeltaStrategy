import data_collector as dc
from kite_auth import ensure_token
import datetime as dt
from kiteconnect import KiteConnect
import config

def debug_strikes_fetching():
    access_token = ensure_token()
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    kite.set_access_token(access_token)
    
    date = dt.date(2026, 3, 23)
    spot = 22700 # approx
    expiry = dt.date(2026, 3, 24)
    
    print(f"--- Debugging Strike Fetching for {date} (Expiry {expiry}) ---")
    
    instruments = kite.instruments("NFO")
    import pandas as pd
    df = pd.DataFrame(instruments)
    
    expiry_str = expiry.strftime("%Y-%m-%d")
    print(f"Filtering for Expiry: {expiry_str}")
    
    nifty_opts = df[
        (df["name"] == "NIFTY") &
        (df["expiry"].astype(str) == expiry_str)
    ]
    
    print(f"Found {len(nifty_opts)} NIFTY instruments for {expiry_str}")
    if not nifty_opts.empty:
        print("Types available:", nifty_opts["instrument_type"].unique())
        print("Strikes present (first 5):", nifty_opts["strike"].unique()[:5])

if __name__ == "__main__":
    debug_strikes_fetching()
