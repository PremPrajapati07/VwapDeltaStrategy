import os
import logging
import datetime as dt
from dotenv import load_dotenv, set_key

# Load config to get API keys if not in env
import config

logging.basicConfig(level=logging.INFO)

def debug():
    print("--- Kite API Verbose Debug ---")
    load_dotenv(override=True)
    
    api_key = os.environ.get("KITE_API_KEY", "")
    token   = os.environ.get("KITE_ACCESS_TOKEN", "")
    
    print(f"DEBUG: api_key: '{api_key}'")
    print(f"DEBUG: token:   '{token}' (Length: {len(token)})")
    
    if token.startswith("'") or token.endswith("'") or token.startswith('"'):
        print("⚠️  CRITICAL: Token in .env has literal quotes! Removing them for this test...")
        token = token.strip("'").strip('"')

    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    
    print("\n1. Testing kite.profile()...")
    try:
        profile = kite.profile()
        print(f"   ✅ SUCCESS! User: {profile.get('user_id')} - {profile.get('user_name')}")
    except Exception as e:
        print(f"   ❌ FAILED Profile: {e}")
        # If profile fails, wait and try to see if it's a specific PERMISSION error
        if "Permission" in str(e):
            print("   (This usually means the API Key or Token is restricted or mismatched)")

    print("\n2. Testing kite.quote('NSE:NIFTY 50')...")
    try:
        q = kite.quote("NSE:NIFTY 50")
        print(f"   ✅ SUCCESS! Price: {q['NSE:NIFTY 50']['last_price']}")
    except Exception as e:
        print(f"   ❌ FAILED Quote: {e}")

    print("\n3. Testing kite.margins()...")
    try:
        m = kite.margins()
        print(f"   ✅ SUCCESS! Available cash: {m['equity']['available']['cash']}")
    except Exception as e:
        print(f"   ❌ FAILED Margins: {e}")

if __name__ == "__main__":
    debug()
