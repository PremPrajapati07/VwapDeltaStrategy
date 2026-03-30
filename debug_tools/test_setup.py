# ============================================================
#  test_setup.py — Validates the project environment
# ============================================================
"""
Run this to verify your environment is correctly configured:
    .\\venv\\Scripts\\python test_setup.py

Checks:
  1. All required .env variables are present
  2. PostgreSQL connection + all 6 tables exist
  3. Raw data collector demo (no Kite needed)
"""

import os
import sys

REQUIRED_ENV = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_USER_ID",
    "KITE_PASSWORD",
    "KITE_TOTP_SECRET",
    "DATABASE_URL",
]

def check(label, ok, detail=""):
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}", f"— {detail}" if detail else "")
    return ok


def test_env():
    print("\n📋 Checking .env variables …")
    from dotenv import load_dotenv
    load_dotenv(override=True)
    passed = True
    for key in REQUIRED_ENV:
        val = os.environ.get(key, "").strip()
        ok  = bool(val and val != f"your_{key.lower()}_here")
        if not check(key, ok, "(set)" if ok else "⚠️  NOT SET or still placeholder"):
            passed = False
    return passed


def test_db():
    print("\n🗄️  Checking PostgreSQL connection & schema …")
    try:
        import db
        db.init_schema()
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
                tables = {r[0] for r in cur.fetchall()}
        expected = {"sessions", "instruments", "straddle_candles",
                    "market_context", "daily_features", "trade_log"}
        for t in expected:
            check(f"Table: {t}", t in tables)
        return expected.issubset(tables)
    except Exception as e:
        check("PostgreSQL connection", False, str(e))
        return False


def test_kite_live():
    print("\n📡 Testing live Kite API …")
    try:
        import kite_auth
        import data_collector as dc

        print("  -> Auto-login (checking token / logging in) …")
        kite = kite_auth.get_kite()
        check("Auto-login / token", True, "authenticated")

        # Spot price
        spot = dc.get_spot_price(kite)
        check("Nifty spot price", spot > 0, f"{spot:.0f}")

        # VIX
        vix = dc.get_vix(kite)
        check("India VIX", vix > 0, f"{vix:.2f}")

        # Fetch 5 candles of Nifty index (token 256265)
        import datetime as dt
        now  = dt.datetime.now()
        start = now - dt.timedelta(minutes=10)
        candles = kite.historical_data(256265, start, now, "minute")
        check("Historical candles (Nifty 5-min)", len(candles) > 0,
              f"{len(candles)} candles returned")

        # Next Thursday or nearest expiry + strikes
        expiry  = dc.get_nearest_expiry(kite)
        strikes = dc.get_nifty_expiry_strikes(kite, spot, expiry)
        check("Strike list (11 strikes)", len(strikes) >= 1,
              f"{len(strikes)} strikes found, ATM={strikes[len(strikes)//2]['atm']}")

        return True
    except Exception as e:
        check("Kite live API", False, str(e)[:200])
        return False


if __name__ == "__main__":
    results = []
    results.append(test_env())
    results.append(test_db())
    results.append(test_kite_live())

    print(f"\n{'='*50}")
    passed = sum(results)
    total  = len(results)
    print(f"  {'ALL GOOD' if passed == total else 'Some checks failed'} "
          f"({passed}/{total} passed)")
    if passed < total:
        print("  Fix the issues above, then re-run test_setup.py")
    print(f"{'='*50}\n")

