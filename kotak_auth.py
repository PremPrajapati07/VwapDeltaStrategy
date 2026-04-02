# ============================================================
#  kotak_auth.py — Fully Automated Kotak Neo Login
#  Supports both SDK versions:
#    v1 (kotak-neo-api):    NeoAPI(consumer_key, consumer_secret) → login() → session_2fa()
#    v2 (Kotak-neo-api-v2): NeoAPI(consumer_key only)             → totp_login() → totp_validate()
# ============================================================
import inspect
import logging
import time
import pyotp
from neo_api_client import NeoAPI
import config

log = logging.getLogger(__name__)

# Module-level singleton — reuse the same authenticated client for the whole session
_neo_client = None


def _detect_sdk_version() -> str:
    sig = inspect.signature(NeoAPI.__init__)
    return "v2" if "consumer_secret" not in list(sig.parameters.keys()) else "v1"


def get_neo_client(force_relogin: bool = False):
    """
    Returns an authenticated NeoAPI instance.
    Reuses the existing session if already logged in (avoids redundant logins).
    Pass force_relogin=True to force a fresh login (e.g. after 'unauthorized').
    """
    global _neo_client

    if not config.ENABLE_KOTAK:
        log.info("Kotak Neo is disabled in config.")
        return None

    # Return existing live session without re-logging in
    if _neo_client and not force_relogin:
        if validate_session(_neo_client):
            log.info("Reusing existing Kotak Neo session.")
            return _neo_client
        else:
            log.warning("Existing Neo session is dead. Re-logging in...")

    missing = [
        name for name, val in [
            ("NEO_CONSUMER_KEY",  config.NEO_CONSUMER_KEY),
            ("NEO_MOBILE_NUMBER", config.NEO_MOBILE_NUMBER),
            ("NEO_MPIN",          config.NEO_MPIN),
            ("NEO_TOTP_SECRET",   config.NEO_TOTP_SECRET),
        ] if not val
    ]
    if missing:
        raise ValueError(f"Missing Kotak credentials in .env: {', '.join(missing)}")

    sdk = _detect_sdk_version()
    log.info(f"Detected Kotak SDK: {sdk}")

    for attempt in range(3):
        try:
            mobile    = str(config.NEO_MOBILE_NUMBER).strip()[-10:]
            # Generate a FRESH TOTP at login time — do NOT reuse across attempts
            totp_code = pyotp.TOTP(config.NEO_TOTP_SECRET.strip()).now()
            log.info(f"Login attempt {attempt+1}/3 | TOTP: {totp_code}")

            client = _login_v2(mobile, totp_code) if sdk == "v2" else _login_v1(mobile, totp_code)

            # Give the session a moment to activate server-side
            time.sleep(2)

            if validate_session(client):
                print("   ✅ Kotak Neo session validated.")
                log.info("✅ Kotak Neo session validated successfully.")
                _neo_client = client
                return client
            else:
                log.warning(f"Session validation failed on attempt {attempt+1}. "
                             f"Waiting 5s before retry...")
                time.sleep(5)

        except Exception as e:
            log.error(f"Login attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(3)

    log.error("All 3 Kotak Neo login attempts failed.")
    print("❌ [ERROR] Kotak Neo login failed after 3 attempts. Check credentials and network.")
    return None


def validate_session(client) -> bool:
    """
    Verifies the Kotak Neo session is alive.
    Accepts ANY non-error response from limits() as valid — Neo returns many
    different response shapes depending on SDK version.
    """
    try:
        res = client.limits()
        res_str = str(res).strip().lower()

        # Explicit failure signals
        if res_str in ("unauthorized", "none", "", "not_ok"):
            log.warning(f"validate_session: explicit failure response: {res!r}")
            return False
        if isinstance(res, dict) and (res.get("stat", "").lower() == "not_ok" or res.get("id") == "10020"):
            log.warning(f"validate_session: authentication failed (10020): {res}")
            return False

        # Any other response (dict with data, dict without data, non-empty string)
        # means the session is alive — Neo is inconsistent about response shape.
        log.info(f"validate_session: session alive. Response type={type(res).__name__} "
                 f"keys={list(res.keys()) if isinstance(res, dict) else 'N/A'}")
        return True

    except Exception as e:
        log.warning(f"validate_session exception: {e}")
        return False


def refresh_if_needed(client) -> object:
    """
    Called before placing orders. If the session is dead, re-login and return
    a fresh client. Returns the (possibly new) client.
    Used by NeoBroker to auto-recover from 'unauthorized' without crashing.
    """
    if validate_session(client):
        return client
    log.warning("Neo session expired — auto-refreshing before order...")
    print("   🔄 Neo session expired. Auto re-logging in...")
    return get_neo_client(force_relogin=True)


# ── v2 Login Flow ─────────────────────────────────────────────
def _login_v2(mobile: str, totp_code: str):
    log.info("Using v2 login flow (totp_login - totp_validate)...")

    client = NeoAPI(
        environment=str(config.NEO_ENVIRONMENT).upper(),
        consumer_key=config.NEO_CONSUMER_KEY,
    )

    print(f"   🔑 Logging in via UCC {config.NEO_USER_ID}...")
    login_resp = client.totp_login(
        mobile_number=f"+91{mobile}",
        ucc=config.NEO_USER_ID,
        totp=totp_code,
    )
    log.info(f"totp_login FULL RESPONSE: {login_resp}")
    _assert_no_error(login_resp, "totp_login")

    time.sleep(1)
    print("   🔐 Validating TOTP Session with MPIN...")
    validate_resp = client.totp_validate(mpin=config.NEO_MPIN)
    log.info(f"totp_validate FULL RESPONSE: {validate_resp}")
    # Capture baseUrl and token from validate_resp
    data = validate_resp.get("data", {})
    token = data.get("token", "")
    baseUrl = data.get("baseUrl", "").strip("/") # Cluster URL (e.g. e21.kotaksecurities.com)
    
    # ── Diagnostic Logs (User Requested) ──
    print(f"   LOGIN TOKEN: {token[:20]}...")
    print(f"   LOGIN BASE URL: {baseUrl}")
    log.info(f"Login Diagnostics: baseUrl={baseUrl} token={token[:15]}...")

    # ── Critical Fix: Monkey Patch SDK to support Cluster URLs ──
    # The NeoAPI SDK hardcodes a check for 'prod'/'uat' in get_domain(). 
    # If we set host to 'e21.kotaksecurities.com', it crashes with ApiValueError.
    # By patching get_domain, we keep host='PROD' but force the URL to the cluster.
    if baseUrl:
        try:
            from types import MethodType
            
            # The cluster URL from Kotak often needs a trailing slash for the SDK's join logic
            final_cluster_url = baseUrl if baseUrl.endswith("/") else (baseUrl + "/")

            def patched_get_domain(self_utility, session_init=False):
                # Always return our cluster URL instead of the default generic one
                log.info(f"Using Patched Cluster Domain: {final_cluster_url}")
                return final_cluster_url

            # Bind the new function to the specific instance
            client.configuration.get_domain = MethodType(patched_get_domain, client.configuration)
            log.info(f"Successfully monkey-patched NeoUtility.get_domain to {final_cluster_url}")
            
            # Also ensure the generic host string is valid to pass the SDK's internal 'in host_list' check
            client.configuration.host = "PROD" 

        except Exception as e:
            log.warning(f"Could not monkey-patch SDK domain: {e}")

    # Finalize session for trading
    time.sleep(3) 
    log.info("✅ Kotak Neo v2 login successful with Cluster Redirection.")
    return client


# ── v1 Login Flow ─────────────────────────────────────────────
def _login_v1(mobile: str, totp_code: str):
    log.info("Using v1 login flow (login → session_2fa)...")

    if not config.NEO_CONSUMER_SECRET:
        raise ValueError("NEO_CONSUMER_SECRET is required for kotak-neo-api v1 SDK.")

    client = NeoAPI(
        consumer_key=config.NEO_CONSUMER_KEY,
        consumer_secret=config.NEO_CONSUMER_SECRET,
        environment=config.NEO_ENVIRONMENT,
        access_token=None,
        neo_fin_key=None,
    )

    log.info(f"Calling login() for mobile: +91{mobile}")
    login_resp = client.login(mobilenumber=f"+91{mobile}", password=totp_code)
    log.info(f"login response: {login_resp}")
    _assert_no_error(login_resp, "login")

    time.sleep(1)
    log.info("Calling session_2fa() with MPIN...")
    fa_resp = client.session_2fa(OTP=config.NEO_MPIN)
    log.info(f"session_2fa response: {fa_resp}")
    _assert_no_error(fa_resp, "session_2fa")

    log.info("✅ Kotak Neo v1 login successful.")
    return client


# ── Shared helper ─────────────────────────────────────────────
def _assert_no_error(resp, step: str):
    if resp is None:
        raise ValueError(f"{step} returned None — check credentials or API status.")
    if isinstance(resp, dict):
        err = resp.get("error") or resp.get("Error") or resp.get("message") if resp.get("success") is False else None
        if err:
            raise ValueError(f"{step} error: {err}")


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    neo = get_neo_client()
    if neo:
        print("✅ Login Success! Testing positions fetch...")
        try:
            pos = neo.positions()
            print(f"   Positions response: {pos}")
        except Exception as e:
            print(f"   Positions fetch error (non-critical): {e}")
    else:
        print("❌ Login Failed.")