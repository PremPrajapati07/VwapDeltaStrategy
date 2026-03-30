import os
import time
from kiteconnect import KiteConnect
from dotenv import load_dotenv

def run_test():
    load_dotenv(override=True)
    api_key = os.environ.get("KITE_API_KEY", "")
    token = os.environ.get("KITE_ACCESS_TOKEN", "").strip("'").strip('"')
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    
    print(f"--- START TEST (API_KEY={api_key[:4]}...) ---")
    
    try:
        print("Step 1: profile()...")
        p = kite.profile()
        print(f"  SUCCESS: {p['user_id']}")
    except Exception as e:
        print(f"  FAILED profile: {e}")
    
    time.sleep(1)
    
    try:
        print("Step 2: quote('NSE:RELIANCE')...")
        q = kite.quote("NSE:RELIANCE")
        print(f"  SUCCESS Reliance: {q['NSE:RELIANCE']['last_price']}")
    except Exception as e:
        print(f"  FAILED quote Reliance: {e}")
        
    time.sleep(1)

    try:
        print("Step 3: quote('NSE:NIFTY 50')...")
        q = kite.quote("NSE:NIFTY 50")
        print(f"  SUCCESS Nifty: {q['NSE:NIFTY 50']['last_price']}")
    except Exception as e:
        print(f"  FAILED quote Nifty: {e}")

    time.sleep(1)
    
    try:
        print("Step 4: instruments('NFO')...")
        ins = kite.instruments("NFO")
        print(f"  SUCCESS: Fetched {len(ins)} instruments for NFO.")
    except Exception as e:
        print(f"  FAILED instruments: {e}")

    time.sleep(1)
    
    try:
        print("Step 5: margins()...")
        m = kite.margins()
        print(f"  SUCCESS: Cash={m['equity']['available']['cash']}")
    except Exception as e:
        print(f"  FAILED margins: {e}")

    print("--- END TEST ---")

if __name__ == "__main__":
    run_test()
