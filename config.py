# ============================================================
#  config.py — Loads all settings from .env
# ============================================================
import os
from dotenv import load_dotenv

# Load .env from the project root using an absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# ── Zerodha Kite Credentials (from .env) ──────────────────────
ENABLE_ZERODHA    = os.environ.get("ENABLE_ZERODHA", "True").lower() == "true"
KITE_API_KEY      = os.environ.get("KITE_API_KEY", "g8oeajvc53f6v58x")
KITE_API_SECRET   = os.environ.get("KITE_API_SECRET", "v3dn3xiat7hmmu8od9p65v5yskrqvhia")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")  # refreshed at runtime

# ── Kotak Neo Credentials (from .env) ───────────────────────
ENABLE_KOTAK      = os.environ.get("ENABLE_KOTAK", "True").lower() == "true"
NEO_CONSUMER_KEY  = os.environ.get("NEO_CONSUMER_KEY", "")
NEO_CONSUMER_SECRET = os.environ.get("NEO_CONSUMER_SECRET", "")
NEO_USER_ID       = os.environ.get("NEO_USER_ID", "")
NEO_MOBILE_NUMBER = os.environ.get("NEO_MOBILE_NUMBER", "")
NEO_PASSWORD      = os.environ.get("NEO_PASSWORD", "")
NEO_MPIN          = os.environ.get("NEO_MPIN", "")
NEO_TOTP_SECRET   = os.environ.get("NEO_TOTP_SECRET", "")
NEO_ENVIRONMENT   = os.environ.get("NEO_ENVIRONMENT", "prod")

# ── Database ───────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:admin@localhost:5433/VwapsDelta24March")

# ── Instrument ─────────────────────────────────────────────────
INDEX_SYMBOL            = "NIFTY"
EXCHANGE                = "NFO"
UNDERLYING              = "NSE:NIFTY 50"   # for spot price fetch
LOT_SIZE                = 65               # default lot size for Nifty
DEFAULT_LOTS            = 1                # default number of lots to trade

# ── Strike Selection ───────────────────────────────────────────
STRIKE_INTERVAL         = 100              # Nifty strike gap (universal 100)
NUM_STRIKES_EACH_SIDE   = 5                # ±5 strikes from ATM → 11 total
SCAN_TIME               = "10:10"          # time to lock strikes & run ML
VWAP_START_TIME         = "10:15"          # VWAP calculation start

# ── Entry / Exit Rules ─────────────────────────────────────────
ENTRY_CONDITION             = "below_vwap"   # sell when straddle < VWAP
EXIT_CONDITION              = "above_vwap"   # buy back when straddle > VWAP
HARD_EXIT_TIME              = "15:00"        # force close all positions

# ── Model Arjun Settings (Warrior) ─────────────────────────────
ARJUN_EXIT_THRESHOLD        = 0.55           # Optimized for 2025 (Lower = more reactive)

# ── Strategy Guardrails (Optimized for "Profit King" 8,251 pts) 
KRISHNA_MIN_CONFIDENCE       = 0.0           # Trade every daily strike chosen by Krishna
DAILY_STOP_LOSS              = -100.0        # Original safety net (8,251 pt config)
DAILY_PROFIT_TARGET          = 9999.0        # No limit - captures huge trend moves
MAX_TRADES_FOR_PROFIT_TARGET = 5             # Let the strategy trade all day
MAX_TRADES_PER_DAY           = 5             # 5 max entries
SL_COOLDOWN_MINUTES          = 15            # Cooldown after SL hit

# ── Risk & Reliability ─────────────────────────────────────────
REENTRY_ALLOWED             = True
REENTRY_REEVALUATE_STRIKE   = True           # re-run ML on re-entry
SL_MULTIPLIER               = 2.0
CHECK_MARGIN                = False          # check available cash before entry
MIN_REQUIRED_MARGIN         = 150000         # minimum cash required to trade (approx 1 straddle)
WAIT_FOR_MARKET_OPEN        = True           # wait until 9:15 AM in live mode

# ── File Paths (logs & models) ───────────────────────────────
LOGS_PATH   = "logs/"
MODELS_PATH = "models/"
STATE_FILE  = "logs/trading_state.json"

# ── Backtest Defaults ──────────────────────────────────────────
BACKTEST_START = "2023-01-01"
BACKTEST_END   = "2024-12-31"

# ── ML Model Features (Model Krishna) ─────────────────────────
ML_FEATURES = [
    "vwap_gap_pct", "straddle_premium", "ce_pe_ratio", "distance_from_atm",
    "distance_from_atm_pct", "oi_ce", "oi_pe", "oi_imbalance", "days_to_expiry",
    "day_of_week", "vix", "nifty_prev_day_change", "nifty_open_gap", "ce_iv",
    "pe_iv", "ce_delta", "pe_delta", "ce_theta", "pe_theta", "ce_gamma",
    "pe_gamma", "ce_vega", "pe_vega", "delta_spread", "theta_ratio", "vega_imbalance"
]
