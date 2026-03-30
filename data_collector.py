# ============================================================
#  data_collector.py — Historical & Live Data via Kite Connect
# ============================================================
"""
Responsibilities:
  1. Authenticate with Zerodha Kite Connect (via kite_auth)
  2. Fetch per-minute OHLCV for CE + PE of all 11 strikes
  3. Fetch India VIX, Nifty spot, prev close
  4. Store raw data in PostgreSQL for backtesting & ML training
  5. Stream live quotes during market hours
"""

import os
import logging
import datetime as dt
import pandas as pd
import time
import requests
import functools
from typing import Optional
from kiteconnect import KiteConnect

import config
import db

os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, "data_collector.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

def retry_api(retries=3, delay=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, Exception) as e:
                    err_str = str(e).lower()
                    if "timed out" in err_str or "connection" in err_str or "timeout" in err_str:
                        if attempt < retries - 1:
                            log.warning(f"Kite API network error inside {func.__name__}: {e}. Retrying {attempt+1}/{retries} in {delay}s...")
                            time.sleep(delay)
                            continue
                    log.error(f"Kite API failed permanently in {func.__name__}: {e}")
                    raise
        return wrapper
    return decorator


# ── Instrument Lookup ────────────────────────────────────────

@retry_api(retries=3, delay=2.0)
def get_nifty_expiry_strikes(kite: KiteConnect, spot: float, expiry: dt.date) -> list[dict]:
    """
    Build list of 11 strikes (ATM ± 5 × 50 pts) for given expiry.
    Returns list of dicts with keys: strike, ce_token, pe_token, ce_symbol, pe_symbol
    """
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)

    atm = round(spot / config.STRIKE_INTERVAL) * config.STRIKE_INTERVAL
    strikes = [
        atm + i * config.STRIKE_INTERVAL
        for i in range(-config.NUM_STRIKES_EACH_SIDE, config.NUM_STRIKES_EACH_SIDE + 1)
    ]

    expiry_str = expiry.strftime("%Y-%m-%d")
    nifty_opts = df[
        (df["name"] == "NIFTY") &
        (df["expiry"].astype(str) == expiry_str) &
        (df["instrument_type"].isin(["CE", "PE"]))
    ]

    result = []
    for strike in strikes:
        ce = nifty_opts[(nifty_opts["strike"] == strike) & (nifty_opts["instrument_type"] == "CE")]
        pe = nifty_opts[(nifty_opts["strike"] == strike) & (nifty_opts["instrument_type"] == "PE")]
        if ce.empty or pe.empty:
            log.warning(f"Missing CE/PE for strike {strike} expiry {expiry_str}")
            continue
        result.append({
            "strike":     strike,
            "ce_token":   int(ce.iloc[0]["instrument_token"]),
            "pe_token":   int(pe.iloc[0]["instrument_token"]),
            "ce_symbol":  ce.iloc[0]["tradingsymbol"],
            "pe_symbol":  pe.iloc[0]["tradingsymbol"],
            "expiry":     expiry_str,
            "atm":        atm,
        })
    log.info(f"Found {len(result)} strikes around ATM {atm} for expiry {expiry_str}")
    return result


def get_nearest_expiry(kite: KiteConnect, symbol: str = "NIFTY", base_date: Optional[dt.date] = None) -> dt.date:
    """
    Find the earliest expiry >= base_date from the active instrument master.
    """
    try:
        instruments = kite.instruments("NFO")
        df = pd.DataFrame(instruments)
        # Filter for the symbol and options
        opts = df[(df["name"] == symbol) & (df["instrument_type"].isin(["CE", "PE"]))]
        expiries = sorted(opts["expiry"].dropna().unique())
        
        target = base_date or dt.date.today()
        future_expiries = [e for e in expiries if e >= target]
        
        if future_expiries:
            log.info(f"Dynamic Expiry: Nearest {symbol} expiry for {target} is {future_expiries[0]}")
            return future_expiries[0]
            
        raise ValueError(f"No future expiries found for {symbol} after {target}")
    except Exception as e:
        log.error(f"Failed to find nearest expiry: {e}")
        # Fallback to next Thursday
        d = base_date or dt.date.today()
        days_ahead = (3 - d.weekday()) % 7
        return d + dt.timedelta(days=days_ahead)


def get_next_thursday(from_date: Optional[dt.date] = None) -> dt.date:
    """Legacy helper for backfill: returns next Thursday from from_date."""
    d = from_date or dt.date.today()
    days_ahead = (3 - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


@retry_api()
def get_spot_price(kite: KiteConnect) -> float:
    """Fetch current Nifty 50 spot price."""
    quote = kite.quote(config.UNDERLYING)
    return quote[config.UNDERLYING]["last_price"]


@retry_api()
def get_vix(kite: KiteConnect) -> float:
    """Fetch India VIX."""
    try:
        q = kite.quote("NSE:INDIA VIX")
        return q["NSE:INDIA VIX"]["last_price"]
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return 0.0


@retry_api()
def get_nifty_prev_close(kite: KiteConnect, date: dt.date) -> float:
    """Fetch Nifty 50 closing price of the previous trading day."""
    try:
        # Fetch last 7 days to ensure we catch a trading day across weekends/holidays
        start = date - dt.timedelta(days=7)
        end   = date - dt.timedelta(days=1)
        candles = kite.historical_data(256265, start, end, "day")
        if candles:
            return candles[-1]["close"]
    except Exception as e:
        log.warning(f"Failed to fetch Nifty prev close for {date}: {e}")
    return 0.0


@retry_api()
def get_nifty_open(kite: KiteConnect, date: dt.date) -> float:
    """Fetch Nifty 50 open price at 9:15 AM on a specific date."""
    try:
        start = dt.datetime.combine(date, dt.time(9, 15))
        end   = dt.datetime.combine(date, dt.time(9, 16))
        candles = kite.historical_data(256265, start, end, "minute")
        if candles:
            return candles[0]["open"]
    except Exception as e:
        log.warning(f"Failed to fetch Nifty open for {date}: {e}")
    return 0.0


def save_market_context(kite: KiteConnect, date: dt.date, expiry: dt.date, atm: int, spot: float, prev_close: float, vix: float):
    """Save daily context (VIX, gaps) to PostgreSQL."""
    open_p = get_nifty_open(kite, date)
    open_price = open_p if open_p > 0 else spot
    
    open_gap    = (open_price - prev_close) if (open_price and prev_close) else 0.0
    prev_change = ((spot - prev_close) / prev_close) if (spot and prev_close) else 0.0

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO market_context 
                (trade_date, nifty_spot_920, nifty_prev_close, nifty_prev_day_change, nifty_open_gap, vix, expiry, atm)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_date) DO UPDATE SET
                    nifty_spot_920 = EXCLUDED.nifty_spot_920,
                    nifty_prev_close = EXCLUDED.nifty_prev_close,
                    nifty_prev_day_change = EXCLUDED.nifty_prev_day_change,
                    nifty_open_gap = EXCLUDED.nifty_open_gap,
                    vix = EXCLUDED.vix,
                    expiry = EXCLUDED.expiry,
                    atm = EXCLUDED.atm
            """, (date, spot, prev_close, prev_change, open_gap, vix, expiry, atm))
        conn.commit()


# ── Historical Data ──────────────────────────────────────────

def fetch_minute_data(kite: KiteConnect, instrument_token: int,
                      from_dt: dt.datetime, to_dt: dt.datetime) -> pd.DataFrame:
    """Fetch 1-minute OHLCV candles for a single instrument."""
    try:
        records = kite.historical_data(
            instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval="minute"
        )
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["instrument_token"] = instrument_token
        return df
    except Exception as e:
        log.error(f"Error fetching token {instrument_token}: {e}")
        return pd.DataFrame()


def fetch_straddle_minute_data(kite: KiteConnect, strike_info: dict,
                               trade_date: dt.date) -> pd.DataFrame:
    """
    Fetch 1-min data for CE + PE of a single strike on a trade date.
    Returns merged DataFrame with straddle_price = ce_close + pe_close.
    """
    start = dt.datetime.combine(trade_date, dt.time(9, 15))
    end   = dt.datetime.combine(trade_date, dt.time(15, 30))

    ce_df = fetch_minute_data(kite, strike_info["ce_token"], start, end)
    pe_df = fetch_minute_data(kite, strike_info["pe_token"], start, end)

    if ce_df.empty or pe_df.empty:
        return pd.DataFrame()

    ce_df = ce_df.rename(columns={
        "open": "ce_open", "high": "ce_high", "low": "ce_low",
        "close": "ce_close", "volume": "ce_volume"
    })
    pe_df = pe_df.rename(columns={
        "open": "pe_open", "high": "pe_high", "low": "pe_low",
        "close": "pe_close", "volume": "pe_volume"
    })

    merged = pd.merge(
        ce_df[["date", "ce_open", "ce_high", "ce_low", "ce_close", "ce_volume"]],
        pe_df[["date", "pe_open", "pe_high", "pe_low", "pe_close", "pe_volume"]],
        on="date", how="inner"
    )

    merged["strike"]           = strike_info["strike"]
    merged["expiry"]           = strike_info["expiry"]
    merged["atm"]              = strike_info["atm"]
    merged["trade_date"]       = trade_date.isoformat()
    merged["straddle_price"]   = merged["ce_close"] + merged["pe_close"]
    merged["straddle_volume"]  = merged["ce_volume"] + merged["pe_volume"]
    merged["ce_oi"]            = 0   # historical endpoint may not return OI
    merged["pe_oi"]            = 0

    return merged


# ── VWAP Calculation ─────────────────────────────────────────

def compute_straddle_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative VWAP on straddle from 9:15.
    VWAP = cumsum(straddle_price × straddle_volume) / cumsum(straddle_volume)
    """
    df = df.sort_values("date").copy()
    df["cum_pv"]       = (df["straddle_price"] * df["straddle_volume"]).cumsum()
    df["cum_vol"]      = df["straddle_volume"].cumsum()
    df["vwap"]         = df["cum_pv"] / df["cum_vol"].replace(0, float("nan"))
    df["vwap_gap"]     = df["straddle_price"] - df["vwap"]
    df["vwap_gap_pct"] = df["vwap_gap"] / df["vwap"] * 100
    return df


# ── PostgreSQL Storage ────────────────────────────────────────

def save_candles(df: pd.DataFrame):
    """Insert straddle candles into PostgreSQL, skip duplicates."""
    if df.empty:
        return
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO straddle_candles
                    (trade_date, expiry, strike, atm, ts,
                     ce_open, ce_high, ce_low, ce_close, ce_volume, ce_oi,
                     pe_open, pe_high, pe_low, pe_close, pe_volume, pe_oi,
                     straddle_price, straddle_volume, vwap, vwap_gap_pct)
                    VALUES
                    (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s)
                    ON CONFLICT (trade_date, strike, ts) DO NOTHING
                """, (
                    row["trade_date"],
                    row["expiry"],
                    int(row["strike"]),
                    int(row["atm"]),
                    row["date"],
                    row.get("ce_open"),  row.get("ce_high"),
                    row.get("ce_low"),   row["ce_close"],
                    int(row.get("ce_volume", 0)), int(row.get("ce_oi", 0)),
                    row.get("pe_open"),  row.get("pe_high"),
                    row.get("pe_low"),   row["pe_close"],
                    int(row.get("pe_volume", 0)), int(row.get("pe_oi", 0)),
                    row["straddle_price"],
                    int(row.get("straddle_volume", 0)),
                    row.get("vwap"),
                    row.get("vwap_gap_pct"),
                ))


def get_nifty_open(kite: KiteConnect, trade_date: dt.date) -> float:
    """Fetch Nifty 50 open price at 9:15 AM on trade_date."""
    try:
        start = dt.datetime.combine(trade_date, dt.time(9, 15))
        end   = dt.datetime.combine(trade_date, dt.time(9, 16))
        candles = kite.historical_data(256265, start, end, "minute")
        if candles:
            return candles[0]["open"]
    except Exception as e:
        log.warning(f"9:15 open fetch failed: {e}")
    return 0.0


def save_market_context(kite: KiteConnect, trade_date: dt.date, expiry: dt.date, atm: int,
                        spot_920: float, prev_close: float,
                        vix: float):
    """Upsert daily market context (VIX, gaps) into market_context table."""
    nifty_open = get_nifty_open(kite, trade_date)
    
    prev_change = round((spot_920 - prev_close) / prev_close * 100, 4) if prev_close else 0.0
    open_gap    = round((nifty_open - prev_close) / prev_close * 100, 4) if prev_close and nifty_open else 0.0

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO market_context
                (trade_date, expiry, atm, nifty_spot_920, nifty_prev_close,
                 nifty_prev_day_change, nifty_open_gap, vix)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (trade_date) DO UPDATE
                    SET nifty_spot_920        = EXCLUDED.nifty_spot_920,
                        nifty_prev_close      = EXCLUDED.nifty_prev_close,
                        nifty_prev_day_change = EXCLUDED.nifty_prev_day_change,
                        nifty_open_gap        = EXCLUDED.nifty_open_gap,
                        vix                   = EXCLUDED.vix
            """, (trade_date, expiry, atm, spot_920, prev_close,
                  prev_change, open_gap, vix))


# ── Full Historical Backfill ─────────────────────────────────

def backfill_history(kite: KiteConnect,
                     start_date: str = config.BACKTEST_START,
                     end_date:   str = config.BACKTEST_END):
    """
    For every Thursday between start_date and end_date:
      - Fetch spot at 9:20 → compute ATM → get 11 strikes
      - Fetch VIX + prev close → save market_context
      - Fetch 1-min straddle data for all 11 strikes
      - Compute VWAP  →  Save to PostgreSQL
    """
    db.init_schema()
    start = dt.date.fromisoformat(start_date)
    end   = dt.date.fromisoformat(end_date)

    all_days = [
        start + dt.timedelta(days=i)
        for i in range((end - start).days + 1)
    ]
    # Filter for weekdays (Mon-Fri) if starting from historical archive, 
    # but for Kite we usually only want days with data.
    trading_days = [d for d in all_days if d.weekday() < 5]

    log.info(f"Backfilling {len(trading_days)} days from {start_date} to {end_date}")
    print(f"📦 Backfilling {len(trading_days)} days …\n")

    nifty_token = 256265
    vix_token   = 264969  # India VIX

    for trade_date in trading_days:
        log.info(f"Processing {trade_date} …")
        print(f"  📅 {trade_date}", end="  ")
        try:
            # 1. 9:20 AM Spot for Strike Selection
            scan_start  = dt.datetime.combine(trade_date, dt.time(9, 19))
            scan_end    = dt.datetime.combine(trade_date, dt.time(9, 21))
            spot_candles = kite.historical_data(nifty_token, scan_start, scan_end, "minute")
            if not spot_candles:
                print("⚠️  No spot data, skipping.")
                continue
            spot = spot_candles[-1]["close"]

            # 2. Expiry & Strikes (uses config.STRIKE_INTERVAL=100)
            expiry  = get_nearest_expiry(kite, base_date=trade_date)
            strikes = get_nifty_expiry_strikes(kite, spot, expiry)
            if not strikes:
                print(f"⚠️  No strikes found ({expiry}), skipping.")
                continue
            atm = strikes[len(strikes)//2]["strike"]

            # 3. Market Context
            prev_close = get_nifty_prev_close(kite, trade_date)
            # Fetch VIX at 9:20
            vix = 0.0
            try:
                vix_candles = kite.historical_data(vix_token, scan_start, scan_end, "minute")
                if vix_candles:
                    vix = vix_candles[-1]["close"]
            except Exception as e:
                log.warning(f"VIX historical fetch failed for {trade_date}: {e}")
            
            save_market_context(kite, trade_date, expiry, atm, spot, prev_close, vix)

            # 4. Straddle Candles
            for si in strikes:
                df = fetch_straddle_minute_data(kite, si, trade_date)
                if df.empty:
                    continue
                df = compute_straddle_vwap(df)
                save_candles(df)
                time.sleep(0.35)   # Kite rate limit

            print(f"spot={spot:.0f}  atm={atm}  strikes={len(strikes)}  vix={vix:.2f}")
        except Exception as e:
            log.error(f"Failed {trade_date}: {e}")
            print(f"❌ Error: {e}")
        time.sleep(1)

    log.info("Backfill complete.")
    print("\n✅ Backfill complete.")


# ── Live Snapshot (9:20 AM) ──────────────────────────────────

@retry_api(retries=3, delay=1.0)
def get_live_snapshot(kite: KiteConnect, strikes: list[dict]) -> pd.DataFrame:
    """
    Fetch real-time quote for all CE+PE tokens.
    Returns DataFrame with straddle_price, ce_ltp, pe_ltp, oi_ce, oi_pe per strike.
    """
    tokens = []
    for s in strikes:
        tokens.append(f"NFO:{s['ce_symbol']}")
        tokens.append(f"NFO:{s['pe_symbol']}")

    quotes = kite.quote(tokens)
    rows = []
    for s in strikes:
        ce_q = quotes.get(f"NFO:{s['ce_symbol']}", {})
        pe_q = quotes.get(f"NFO:{s['pe_symbol']}", {})
        ce_ltp = ce_q.get("last_price", 0)
        pe_ltp = pe_q.get("last_price", 0)
        rows.append({
            "strike":          s["strike"],
            "atm":             s["atm"],
            "ce_symbol":       s["ce_symbol"],
            "pe_symbol":       s["pe_symbol"],
            "ce_token":        s["ce_token"],
            "pe_token":        s["pe_token"],
            "ce_ltp":          ce_ltp,
            "pe_ltp":          pe_ltp,
            "straddle_price":  ce_ltp + pe_ltp,
            "oi_ce":           ce_q.get("oi", 0),
            "oi_pe":           pe_q.get("oi", 0),
            "ce_volume":       ce_q.get("volume", 0),
            "pe_volume":       pe_q.get("volume", 0),
        })
    return pd.DataFrame(rows)
