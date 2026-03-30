# ============================================================
#  config.py — Loads all settings from .env
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv(override=True)   # reads .env in project root

# ── Zerodha Kite Credentials (from .env) ──────────────────────
KITE_API_KEY      = os.environ.get("KITE_API_KEY", "g8oeajvc53f6v58x")
KITE_API_SECRET   = os.environ.get("KITE_API_SECRET", "v3dn3xiat7hmmu8od9p65v5yskrqvhia")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")  # refreshed at runtime

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
SCAN_TIME               = "09:20"          # time to lock strikes & run ML
VWAP_START_TIME         = "09:15"          # VWAP calculation start

# ── Entry / Exit Rules ─────────────────────────────────────────
ENTRY_CONDITION             = "below_vwap"   # sell when straddle < VWAP
EXIT_CONDITION              = "above_vwap"   # buy back when straddle > VWAP
HARD_EXIT_TIME              = "15:00"        # force close all positions
REENTRY_ALLOWED             = True
REENTRY_REEVALUATE_STRIKE   = True          # re-run ML on re-entry

# ── Risk & Reliability ─────────────────────────────────────────
SL_MULTIPLIER = 2.0                          # stop loss = 2× premium collected
CHECK_MARGIN  = True                         # check available cash before entry
MIN_REQUIRED_MARGIN = 150000                 # minimum cash required to trade (approx 1 straddle)
SL_COOLDOWN_MINUTES = 15                     # wait 15 mins after SL before re-entry
WAIT_FOR_MARKET_OPEN = True                  # wait until 9:15 AM in live mode

# ── File Paths (logs & models only — no SQLite) ────────────────
MODELS_PATH = "models/"
LOGS_PATH   = "logs/"

# ── Backtest ───────────────────────────────────────────────────
BACKTEST_START = "2023-01-01"
BACKTEST_END   = "2024-12-31"

# ── ML Model Features ──────────────────────────────────────────
ML_FEATURES = [
    "vwap_gap_pct",
    "straddle_premium",
    "ce_pe_ratio",
    "distance_from_atm",
    "distance_from_atm_pct",
    "oi_ce",
    "oi_pe",
    "oi_imbalance",
    "days_to_expiry",
    "day_of_week",
    "vix",
    "nifty_prev_day_change",
    "nifty_open_gap",
]
