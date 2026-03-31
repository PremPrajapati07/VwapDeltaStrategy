# ============================================================
#  db.py — PostgreSQL connection pool + schema initialization
# ============================================================
"""
All database interaction in this project goes through this module.

Usage:
    from db import get_conn, init_schema

    # Context-manager — auto-commits and closes
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

    # One-time at startup
    init_schema()
"""

import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_pool: pool.SimpleConnectionPool = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        url = os.environ["DATABASE_URL"]
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=url,
        )
        log.info("PostgreSQL connection pool created.")
    return _pool


@contextmanager
def get_conn():
    """
    Yields a psycopg2 connection from the pool.
    Auto-commits on clean exit, rolls back on exception, always returns
    the connection to the pool.
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# ── Schema DDL ────────────────────────────────────────────────

_DDL = """

-- Kite session tokens (one per calendar day)
CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    login_date      DATE UNIQUE NOT NULL,
    access_token    TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Nifty instrument master (refreshed daily before backfill)
CREATE TABLE IF NOT EXISTS instruments (
    instrument_token BIGINT PRIMARY KEY,
    tradingsymbol    TEXT NOT NULL,
    strike           INTEGER,
    instrument_type  TEXT,       -- CE / PE
    expiry           DATE,
    name             TEXT,
    lot_size         INTEGER,
    updated_at       TIMESTAMPTZ DEFAULT now()
);

-- 1-min straddle candles (core market data)
CREATE TABLE IF NOT EXISTS straddle_candles (
    id               BIGSERIAL PRIMARY KEY,
    trade_date       DATE NOT NULL,
    expiry           DATE NOT NULL,
    strike           INTEGER NOT NULL,
    atm              INTEGER NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,
    
    -- Leg Prices & Volume
    ce_open          NUMERIC(10,2),
    ce_high          NUMERIC(10,2),
    ce_low           NUMERIC(10,2),
    ce_close         NUMERIC(10,2),
    ce_volume        BIGINT DEFAULT 0,
    ce_oi            BIGINT DEFAULT 0,
    
    pe_open          NUMERIC(10,2),
    pe_high          NUMERIC(10,2),
    pe_low           NUMERIC(10,2),
    pe_close         NUMERIC(10,2),
    pe_volume        BIGINT DEFAULT 0,
    pe_oi            BIGINT DEFAULT 0,
    
    -- Derived Metrics
    straddle_price   NUMERIC(10,2),
    straddle_volume  BIGINT DEFAULT 0,
    vwap             NUMERIC(10,4),
    vwap_gap_pct     NUMERIC(10,4),
    
    -- Synthetic & Greeks (New)
    synthetic_spot   NUMERIC(10,2),
    ce_iv            NUMERIC(10,4),
    pe_iv            NUMERIC(10,4),
    ce_delta         NUMERIC(10,4),
    ce_theta         NUMERIC(10,4),
    ce_gamma         NUMERIC(10,6),
    ce_vega          NUMERIC(10,4),
    pe_delta         NUMERIC(10,4),
    pe_theta         NUMERIC(10,4),
    pe_gamma         NUMERIC(10,6),
    pe_vega          NUMERIC(10,4),

    UNIQUE (trade_date, strike, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_date_strike
    ON straddle_candles (trade_date, strike);
CREATE INDEX IF NOT EXISTS idx_candles_ts
    ON straddle_candles (ts);

-- Market context per day (VIX, spot, gaps)
CREATE TABLE IF NOT EXISTS market_context (
    trade_date              DATE PRIMARY KEY,
    expiry                  DATE,
    atm                     INTEGER,
    nifty_spot_920          NUMERIC(10,2),
    nifty_prev_close        NUMERIC(10,2),
    nifty_prev_day_change   NUMERIC(8,4),
    nifty_open_gap          NUMERIC(8,4),
    vix                     NUMERIC(8,4),
    created_at              TIMESTAMPTZ DEFAULT now()
);

-- Daily ML feature rows (one row per trade_date × strike)
CREATE TABLE IF NOT EXISTS daily_features (
    id                      BIGSERIAL PRIMARY KEY,
    trade_date              DATE NOT NULL,
    expiry                  DATE NOT NULL,
    strike                  INTEGER NOT NULL,
    atm                     INTEGER,
    vwap_gap_pct            NUMERIC(10,4),
    straddle_premium        NUMERIC(10,2),
    ce_pe_ratio             NUMERIC(10,4),
    distance_from_atm       NUMERIC(10,2),
    distance_from_atm_pct   NUMERIC(10,4),
    oi_ce                   BIGINT,
    oi_pe                   BIGINT,
    oi_imbalance            NUMERIC(10,4),
    days_to_expiry          SMALLINT,
    day_of_week             SMALLINT,
    vix                     NUMERIC(8,4),
    nifty_prev_day_change   NUMERIC(8,4),
    nifty_open_gap          NUMERIC(8,4),
    pnl                     NUMERIC(10,2),
    is_best                 BOOLEAN DEFAULT FALSE,
    best_strike             INTEGER,
    UNIQUE (trade_date, strike)
);
CREATE INDEX IF NOT EXISTS idx_features_date
    ON daily_features (trade_date);

-- Trade log (every entry/exit event)
CREATE TABLE IF NOT EXISTS trade_log (
    id               BIGSERIAL PRIMARY KEY,
    trade_date       DATE NOT NULL,
    expiry           DATE NOT NULL,
    selected_strike  INTEGER NOT NULL,
    atm              INTEGER,
    entry_time       TIMESTAMPTZ,
    entry_premium    NUMERIC(10,2),
    exit_time        TIMESTAMPTZ,
    exit_premium     NUMERIC(10,2),
    exit_reason      TEXT,
    pnl              NUMERIC(10,2),
    lots             SMALLINT DEFAULT 1,
    re_entry_count   SMALLINT DEFAULT 0,
    paper_trade      BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_trade_log_date
    ON trade_log (trade_date);

-- Phase 2: Arjun Model training data (Sequential)
CREATE TABLE IF NOT EXISTS arjun_training_data (
    id                 BIGSERIAL PRIMARY KEY,
    trade_date         DATE NOT NULL,
    ts                 TIMESTAMPTZ NOT NULL,
    strike             INTEGER NOT NULL,
    pnl_pts            NUMERIC(10,2),
    max_pnl_so_far     NUMERIC(10,2),
    drawdown           NUMERIC(10,2),
    vwap_gap_pct       NUMERIC(10,4),
    delta_drift        NUMERIC(10,4),
    theta_velocity     NUMERIC(10,4),
    iv_drift           NUMERIC(10,4),
    rel_vol_15m        NUMERIC(10,4),
    should_exit        BOOLEAN DEFAULT FALSE,
    UNIQUE (trade_date, ts, strike)
);

-- Krishna Prediction Log (what the model picked and why)
CREATE TABLE IF NOT EXISTS krishna_predictions (
    id                BIGSERIAL PRIMARY KEY,
    trade_date        DATE UNIQUE NOT NULL,
    predicted_strike  INTEGER NOT NULL,
    confidence        NUMERIC(5,4),
    features_json     JSONB,
    actual_best_strike INTEGER,
    prediction_correct BOOLEAN,
    created_at        TIMESTAMPTZ DEFAULT now()
);

"""


def init_schema():
    """
    Create all tables and indexes if they don't already exist.
    Safe to call on every startup (idempotent).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    log.info("Database schema initialized.")
    print("✅ PostgreSQL schema ready.")


def save_session(login_date, access_token: str):
    """Upsert today's Kite access token into the sessions table."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (login_date, access_token)
                VALUES (%s, %s)
                ON CONFLICT (login_date) DO UPDATE
                    SET access_token = EXCLUDED.access_token,
                        created_at   = now()
            """, (login_date, access_token))
    log.info(f"Access token saved to DB for {login_date}.")


def load_session(login_date) -> str | None:
    """Return today's access token from DB, or None if not found."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT access_token FROM sessions WHERE login_date = %s",
                (login_date,)
            )
            row = cur.fetchone()
    return row[0] if row else None
