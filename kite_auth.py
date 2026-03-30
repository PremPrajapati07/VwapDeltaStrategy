# ============================================================
#  kite_auth.py — Fully Automated Kite Login
# ============================================================
"""
Automates the Zerodha Kite login flow:
  1. Opens headless Chrome via Selenium
  2. Fills User ID + Password
  3. Generates 6-digit TOTP from secret (pyotp)
  4. Captures request_token from the redirect URL
  5. Exchanges it for an access_token via KiteConnect
  6. Saves token to PostgreSQL sessions table + .env file

Public API:
    ensure_token() -> str
        Returns a valid access token for today, logging in if needed.

    get_kite() -> KiteConnect
        Returns a fully authenticated KiteConnect instance.
"""

import os
import re
import datetime as dt
import logging
import time

import pyotp
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

load_dotenv(override=True)
log = logging.getLogger(__name__)

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
KITE_LOGIN_URL = "https://kite.zerodha.com/"


# ── Selenium auto-login ───────────────────────────────────────

def _selenium_get_request_token(api_key: str, user_id: str,
                                 password: str, totp_secret: str) -> str:
    """
    Drives a headless Chrome session through Kite's login flow.
    Returns the request_token string captured from the redirect URL.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as e:
        raise ImportError(
            f"Selenium dependencies missing: {e}\n"
            "Run: pip install selenium webdriver-manager"
        ) from e

    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--log-level=3")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    wait = WebDriverWait(driver, 20)

    try:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        log.info(f"Navigating to Kite login: {login_url}")
        driver.get(login_url)

        # ── Step 1: User ID + Password ──
        try:
            wait.until(EC.presence_of_element_located((By.ID, "userid")))
            driver.find_element(By.ID, "userid").send_keys(user_id)
            driver.find_element(By.ID, "password").send_keys(password)
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
        except Exception as e:
            driver.save_screenshot("login_error_step1.png")
            raise RuntimeError(f"Failed at UserID/Password step. Screenshot saved. Error: {e}")

        time.sleep(5)
        # ── Step 2: TOTP ──
        try:
            # Zerodha sometimes shows "Invalid user ID or password" here
            error_msg = driver.find_elements(By.CLASS_NAME, "error")
            if error_msg and error_msg[0].is_displayed():
                driver.save_screenshot("login_error_creds.png")
                raise ValueError(f"Kite Login Error: {error_msg[0].text}")

            # Wait for any input field to be visible in the TOTP form
            time.sleep(2)
            wait.until(EC.visibility_of_element_located((By.XPATH, "//form//input")))
            time.sleep(2)
            
            # Filter all inputs to find the strictly visible one (the actual TOTP field)
            visible_inputs = [inp for inp in driver.find_elements(By.XPATH, "//form//input") if inp.is_displayed()]
            if not visible_inputs:
                raise ValueError("Could not find a visible TOTP input field.")
            
            time.sleep(2)
            totp_input = visible_inputs[-1] # take the last visible one if there are multiple

            totp = pyotp.TOTP(totp_secret).now()
            log.info(f"Generated TOTP: {totp}")
            
            time.sleep(2)
            # Click and fill all at once. Character-by-character causes StaleElement on React re-renders.
            totp_input.click()
            time.sleep(0.3)
            totp_input.clear()
            try:
                totp_input.send_keys(totp)
            except Exception as e:
                # If the DOM re-renders or auto-submits mid-typing, it will throw a StaleElement error.
                # We can safely ignore this because it means the form is already processing the input.
                log.info(f"Ignored expected staleness during TOTP entry: {e}")
            
            # Click Continue or wait for auto-submit
            time.sleep(1)
            try:
                # Find all active submit buttons and click the visible one
                btns = [btn for btn in driver.find_elements(By.CSS_SELECTOR, "button[type='submit']") if btn.is_displayed() and btn.isEnabled()]
                if btns:
                    btns[0].click()
            except Exception:
                pass
        except Exception as e:
            if not isinstance(e, ValueError):
                driver.save_screenshot("login_error_step2.png")
            raise e

        # ── Step 3: Wait for redirect and capture token ──
        log.info("Waiting for redirect to capture request_token …")
        for _ in range(30):
            url = driver.current_url
            if "request_token=" in url:
                match = re.search(r"request_token=([a-zA-Z0-9]+)", url)
                if match:
                    token = match.group(1)
                    log.info(f"Captured request_token: {token[:8]}…")
                    return token
            
            # Check for common post-login errors (e.g. account locked)
            error_msg = driver.find_elements(By.CLASS_NAME, "error")
            if error_msg and error_msg[0].is_displayed():
                driver.save_screenshot("login_error_final.png")
                raise ValueError(f"Kite Post-Login Error: {error_msg[0].text}")
                
            time.sleep(1)

        driver.save_screenshot("login_timeout.png")
        raise TimeoutError(
            f"request_token not found in URL after TOTP. Current URL: {driver.current_url}. Screenshot saved."
        )
    finally:
        driver.quit()


# ── Token persistence helpers ─────────────────────────────────

def _load_env_token() -> str | None:
    """Return token from .env if set (non-empty)."""
    t = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    return t if t else None


def _save_token_to_env(token: str):
    """Overwrite KITE_ACCESS_TOKEN in the .env file."""
    if os.path.exists(ENV_FILE):
        set_key(ENV_FILE, "KITE_ACCESS_TOKEN", token)
    os.environ["KITE_ACCESS_TOKEN"] = token


def _is_token_valid(kite: KiteConnect) -> bool:
    """Quick API call to check if the access token is still valid."""
    try:
        kite.profile()
        return True
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────

def ensure_token() -> str:
    """
    Returns a valid Kite access token for today.

    Flow:
      1. Check DB for today's token  →  validate  →  return if valid
      2. Check .env KITE_ACCESS_TOKEN  →  validate  →  return if valid
      3. Perform full auto-login via Selenium + TOTP  →  save + return
    """
    import db   # imported here to avoid circular imports at module level

    today     = dt.date.today()
    api_key   = os.environ["KITE_API_KEY"]
    api_secret = os.environ["KITE_API_SECRET"]

    # ── 1. Try DB ──
    log.info("Checking DB for today's access token …")
    db_token = db.load_session(today)
    if db_token:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(db_token)
        if _is_token_valid(kite):
            log.info("Reusing valid token from DB.")
            _save_token_to_env(db_token)
            return db_token
        log.warning("DB token invalid or expired.")

    # ── 2. Try .env token ──
    env_token = _load_env_token()
    if env_token:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(env_token)
        if _is_token_valid(kite):
            log.info("Reusing valid token from .env.")
            db.save_session(today, env_token)
            return env_token
        log.warning(".env token invalid or expired.")

    # ── 3. Full auto-login ──
    log.info("Performing automated Kite login …")
    print("🔐 Auto-login: launching headless Chrome …")

    try:
        user_id     = os.environ["KITE_USER_ID"]
        password    = os.environ["KITE_PASSWORD"]
        totp_secret = os.environ["KITE_TOTP_SECRET"]

        request_token = _selenium_get_request_token(
            api_key, user_id, password, totp_secret
        )

        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(request_token, api_secret=api_secret)
        access_token = session["access_token"]

        _save_token_to_env(access_token)
        db.save_session(today, access_token)
        return access_token

    except Exception as e:
        log.error(f"Auto-login failed: {e}")
        print(f"\n❌ [ERROR] Automated login failed: {e}")
        print("\n--- MANUAL LOGIN FALLBACK ---")
        print(f"1. Open this URL: https://kite.trade/connect/login?api_key={api_key}&v=3")
        print("2. Login manually and look at the URL after it redirects.")
        print("3. Copy the 'request_token' from the URL (e.g. request_token=abc12345)")
        
        manual_token = input("\nEnter access_token OR Paste the FULL redirect URL: ").strip()
        
        if "request_token=" in manual_token:
            import re
            match = re.search(r"request_token=([a-zA-Z0-9]+)", manual_token)
            if match:
                rt = match.group(1)
                kite = KiteConnect(api_key=api_key)
                session = kite.generate_session(rt, api_secret=api_secret)
                at = session["access_token"]
                _save_token_to_env(at)
                db.save_session(today, at)
                return at
        elif len(manual_token) > 20: 
            _save_token_to_env(manual_token)
            db.save_session(today, manual_token)
            return manual_token
        
        raise RuntimeError("Manual login failed or aborted.")


def get_kite() -> KiteConnect:
    """
    Returns a fully authenticated KiteConnect instance.
    Calls ensure_token() internally — safe to call every startup.
    """
    load_dotenv(override=True)
    api_key = os.environ["KITE_API_KEY"]
    token   = ensure_token()
    kite    = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    return kite
