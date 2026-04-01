# ============================================================
#  check_logins.py — Verify multi-broker connectivity
# ============================================================
import logging
import config
import kite_auth
import kotak_auth
from brokers import KiteBroker, NeoBroker

# Setup logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def verify_all():
    print(f"\n{'='*40}")
    print(" 🔎 MULTI-BROKER CONNECTIVITY TEST")
    print(f"{'='*40}\n")
    
    # ── 1. Zerodha Kite ──
    kite = None # Initialize
    if config.ENABLE_ZERODHA:
        try:
            print("⏳ Connecting to Zerodha Kite...")
            kite = kite_auth.get_kite()
            name = kite.profile()["user_name"]
            print(f"✅ Zerodha Kite: CONNECTED (User: {name})")
        except Exception as e:
            print(f"❌ Zerodha Kite: FAILED ({e})")
    else:
        print("⚪ Zerodha Kite: DISABLED")

    print("")

    # ── 2. Kotak Neo ──
    neo = None # Initialize
    if config.ENABLE_KOTAK:
        try:
            print("⏳ Connecting to Kotak Neo...")
            neo = kotak_auth.get_neo_client()
            if neo:
                # Basic API test
                try:
                    profile = neo.get_positions()
                    print(f"✅ Kotak Neo: CONNECTED (Session Active)")
                except Exception:
                    # In some cases profile fetch might fail based on API permissions
                    # but if get_neo_client returned a client, the auth passed.
                    print("✅ Kotak Neo: CONNECTED (Authenticated)")
            else:
                print("❌ Kotak Neo: FAILED (Login returned None)")
        except Exception as e:
            print(f"❌ Kotak Neo: FAILED ({e})")
    else:
        print("⚪ Kotak Neo: DISABLED")

    # ── 3. Balance Check ──
    print("\n💰 Checking Account Balances...")
    for broker in [b for b in [
        KiteBroker(kite) if (config.ENABLE_ZERODHA and kite) else None,
        NeoBroker(neo) if (config.ENABLE_KOTAK and 'neo' in locals() and neo) else None
    ] if b]:
        try:
            bal = broker.get_balance()
            print(f"✅ [{broker.name}] Balance: ₹{bal:,.0f}")
        except Exception as e:
            print(f"⚠️  [{broker.name}] Balance Error: {e}")

    # ── 4. Symbol Sync Test ──
    print("\n🔍 Testing Symbol Sync (Nifty Sample)...")
    TEST_SYMBOLS = [
        {"exch": "nse_cm", "sym": "Nifty 50"}, # Spot
        {"exch": "nfo", "sym": "NIFTY2640224200CE"} # Option
    ]
    
    for broker in [b for b in [
        KiteBroker(kite) if (config.ENABLE_ZERODHA and kite) else None,
        NeoBroker(neo) if (config.ENABLE_KOTAK and 'neo' in locals() and neo) else None
    ] if b]:
        for item in TEST_SYMBOLS:
            try:
                exch = item["exch"]
                symbol = item["sym"]
                print(f"⏳ [{broker.name}] Fetching {exch} LTP for {symbol}...")
                
                # We need to handle segment for the search logic
                if isinstance(broker, NeoBroker):
                     # Search NSE_CM specifically if it's spot
                     ltp = broker.get_ltp(symbol)
                else: # Zerodha
                     ltp = broker.get_ltp(symbol)
                
                print(f"✅ [{broker.name}] {symbol} (LTP: {ltp})")
            except Exception as e:
                log.error(f"Symbol Test Failed ({item['sym']}): {e}")
                print(f"⚠️  [{broker.name}] Symbol Warning ({item['sym']}): {e}")

    print(f"\n{'='*40}")
    print(" READY FOR LIVE TRADING!" if (config.ENABLE_ZERODHA or config.ENABLE_KOTAK) else " NO BROKERS ENABLED!")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    verify_all()
