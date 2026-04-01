# ============================================================
#  data_collector.py — Historical & Live Data via Kite Connect
# ============================================================
import logging
import datetime as dt
import pandas as pd
import time
import functools
from typing import Optional
from kiteconnect import KiteConnect

import config
import db
from scripts.black_scholes_utils import calculate_iv, calculate_greeks

log = logging.getLogger(__name__)


def _get_client(kite_or_neo):
    """Automatically unwrap a BaseBroker wrapper to get the raw SDK client."""
    return getattr(kite_or_neo, "client", getattr(kite_or_neo, "kite", kite_or_neo))


def retry_api(retries=3, delay=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    raise
        return wrapper
    return decorator


# ── Instrument Lookup (Zerodha Only) ────────────────────────

@retry_api(retries=3, delay=2.0)
def get_nifty_expiry_strikes(kite: KiteConnect, spot: float, expiry: dt.date) -> list[dict]:
    """Fetch 11 strikes (±5 from ATM) for NIFTY from Zerodha NFO instrument list."""
    client = _get_client(kite)
    atm = round(spot / config.STRIKE_INTERVAL) * config.STRIKE_INTERVAL
    strikes = [
        atm + i * config.STRIKE_INTERVAL
        for i in range(-config.NUM_STRIKES_EACH_SIDE, config.NUM_STRIKES_EACH_SIDE + 1)
    ]
    
    print("   📋 Fetching Zerodha NFO instrument list …")
    instruments = client.instruments("NFO")
    df = pd.DataFrame(instruments)
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
        if not ce.empty and not pe.empty:
            result.append({
                "strike":    strike,
                "ce_token":  int(ce.iloc[0]["instrument_token"]),
                "pe_token":  int(pe.iloc[0]["instrument_token"]),
                "ce_symbol": ce.iloc[0]["tradingsymbol"],
                "pe_symbol": pe.iloc[0]["tradingsymbol"],
                "expiry":    expiry_str,
                "atm":       atm,
            })
    return result


def get_nearest_expiry(kite: KiteConnect, symbol: str = "NIFTY", base_date: Optional[dt.date] = None) -> dt.date:
    """Find the next expiry date for a symbol using Zerodha instrument list."""
    try:
        client = _get_client(kite)
        instruments = client.instruments("NFO")
        df = pd.DataFrame(instruments)
        opts = df[(df["name"] == symbol) & (df["instrument_type"].isin(["CE", "PE"]))]
        expiries = sorted(opts["expiry"].dropna().unique())
        target = base_date or dt.date.today()
        future = [e for e in expiries if e >= target]
        return future[0] if future else get_next_tuesday(base_date)
    except Exception:
        return get_next_tuesday(base_date)


def get_next_tuesday(from_date: Optional[dt.date] = None) -> dt.date:
    d = from_date or dt.date.today()
    days_ahead = (1 - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


# ── Market Data (Zerodha Only) ───────────────────────────────

@retry_api()
def get_spot_price(kite: KiteConnect) -> float:
    """Fetch live Nifty spot price from Zerodha."""
    client = _get_client(kite)
    quote = client.quote(config.UNDERLYING)
    return quote[config.UNDERLYING]["last_price"]


@retry_api()
def get_vix(kite: KiteConnect) -> float:
    """Fetch live India VIX from Zerodha."""
    client = _get_client(kite)
    try:
        q = client.quote("NSE:INDIA VIX")
        return q["NSE:INDIA VIX"]["last_price"]
    except Exception:
        return 0.0


@retry_api()
def get_nifty_prev_close(kite: KiteConnect, date: dt.date) -> float:
    """Fetch previous day's close for Nifty Spot (ID: 256265)."""
    try:
        client = _get_client(kite)
        # Fetch 1 day of daily data from the past week
        candles = client.historical_data(256265, date - dt.timedelta(days=7), date - dt.timedelta(days=1), "day")
        return candles[-1]["close"] if candles else 0.0
    except Exception:
        return 0.0


@retry_api()
def get_nifty_open(kite: KiteConnect, date: dt.date) -> float:
    """Fetch 9:15 AM open price for Nifty Spot (ID: 256265)."""
    try:
        client = _get_client(kite)
        start = dt.datetime.combine(date, dt.time(9, 15))
        end   = dt.datetime.combine(date, dt.time(9, 16))
        candles = client.historical_data(256265, start, end, "minute")
        return candles[0]["open"] if candles else 0.0
    except Exception:
        return 0.0


def save_market_context(kite: KiteConnect, trade_date: dt.date, expiry: dt.date, atm: int, spot_920: float, prev_close: float, vix: float):
    """Calculate Gaps/Changes and save to PostgreSQL market_context table."""
    n_open = get_nifty_open(kite, trade_date)
    px_chg = round((spot_920 - prev_close) / prev_close * 100, 4) if prev_close else 0.0
    op_gap = round((n_open - prev_close) / prev_close * 100, 4) if prev_close and n_open else 0.0
    
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO market_context 
                (trade_date, expiry, atm, nifty_spot_920, nifty_prev_close, nifty_prev_day_change, nifty_open_gap, vix)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (trade_date) DO UPDATE SET
                    nifty_spot_920       = EXCLUDED.nifty_spot_920,
                    nifty_prev_close      = EXCLUDED.nifty_prev_close,
                    nifty_prev_day_change  = EXCLUDED.nifty_prev_day_change,
                    nifty_open_gap         = EXCLUDED.nifty_open_gap,
                    vix                    = EXCLUDED.vix
            """, (trade_date, expiry, atm, spot_920, prev_close, px_chg, op_gap, vix))


# ── Historical Data ──────────────────────────────────────────

def fetch_minute_data(kite: KiteConnect, instrument_token: int, from_dt: dt.datetime, to_dt: dt.datetime) -> pd.DataFrame:
    try:
        client = _get_client(kite)
        records = client.historical_data(instrument_token, from_date=from_dt, to_date=to_dt, interval="minute")
        df = pd.DataFrame(records)
        if not df.empty:
            df["instrument_token"] = instrument_token
        return df
    except Exception:
        return pd.DataFrame()


def fetch_straddle_minute_data(kite: KiteConnect, strike_info: dict, trade_date: dt.date) -> pd.DataFrame:
    """Fetch minute historical candles for both CE and PE and merge them into a straddle."""
    start = dt.datetime.combine(trade_date, dt.time(9, 15))
    end   = dt.datetime.combine(trade_date, dt.time(15, 30))
    
    ce_df = fetch_minute_data(kite, strike_info["ce_token"], start, end)
    pe_df = fetch_minute_data(kite, strike_info["pe_token"], start, end)
    
    if ce_df.empty or pe_df.empty:
        return pd.DataFrame()
        
    ce_df = ce_df.rename(columns=lambda x: f"ce_{x}" if x != "date" else x)
    pe_df = pe_df.rename(columns=lambda x: f"pe_{x}" if x != "date" else x)
    
    merged = pd.merge(ce_df, pe_df, on="date", how="inner")
    merged["strike"]         = strike_info["strike"]
    merged["expiry"]         = strike_info["expiry"]
    merged["trade_date"]     = trade_date.isoformat()
    merged["straddle_price"] = merged["ce_close"] + merged["pe_close"]
    merged["straddle_volume"] = merged["ce_volume"] + merged["pe_volume"]
    
    return merged


def compute_straddle_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative volume-weighted average price (VWAP) for a straddle."""
    df = df.sort_values("date").copy()
    df["cum_pv"] = (df["straddle_price"] * df["straddle_volume"]).cumsum()
    df["cum_vol"] = df["straddle_volume"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, float("nan"))
    df["vwap_gap_pct"] = (df["straddle_price"] - df["vwap"]) / df["vwap"] * 100
    return df


def save_candles(df: pd.DataFrame):
    """Upsert straddle minute candles into straddle_candles table."""
    if df.empty: return
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for _, r in df.iterrows():
                cur.execute("""
                    INSERT INTO straddle_candles 
                    (trade_date, expiry, strike, atm, ts, ce_close, pe_close, straddle_price, vwap, vwap_gap_pct)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, (r["trade_date"], r["expiry"], int(r["strike"]), int(r.get("atm", 0)), 
                      r["date"], r["ce_close"], r["pe_close"], r["straddle_price"], r["vwap"], r["vwap_gap_pct"]))


def backfill_history(kite: KiteConnect, start_date: str = config.BACKTEST_START, end_date: str = config.BACKTEST_END):
    """Main ingestion loop for Nifty Option data."""
    db.init_schema()
    client = _get_client(kite)
    start  = dt.date.fromisoformat(start_date)
    end    = dt.date.fromisoformat(end_date)
    
    trading_days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    trading_days = [d for d in trading_days if d.weekday() < 5] # weekdays only
    
    for trade_date in trading_days:
        try:
            spot_candles = client.historical_data(256265, dt.datetime.combine(trade_date, dt.time(9, 19)), 
                                                   dt.datetime.combine(trade_date, dt.time(9, 21)), "minute")
            if not spot_candles: continue
            spot = spot_candles[-1]["close"]
            expiry = get_nearest_expiry(kite, base_date=trade_date)
            strikes = get_nifty_expiry_strikes(kite, spot, expiry)
            if not strikes: continue
            
            save_market_context(kite, trade_date, expiry, strikes[len(strikes)//2]["strike"], spot, get_nifty_prev_close(kite, trade_date), 0.0)
            
            for si in strikes:
                df = fetch_straddle_minute_data(kite, si, trade_date)
                if not df.empty:
                    save_candles(compute_straddle_vwap(df))
        except Exception:
            continue


# ── Live snapshot & Greeks ──────────────────────────────────

def _add_greeks_to_df(df: pd.DataFrame, spot: float, expiry_str: str) -> pd.DataFrame:
    """Internal helper to calculate IV and Greeks for Zerodha live snapshots."""
    if df.empty or spot <= 0: return df
    try:
        expiry_dt = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = dt.date.today()
        days_to_exp = (expiry_dt - today).days
        # T: Time to expiry in years. 
        T = max(days_to_exp, 0.5) / 365.0
        r = 0.07 

        for idx, row in df.iterrows():
            strike = float(row["strike"])
            # CE Greeks
            ce_iv = calculate_iv(row["ce_ltp"], spot, strike, T, r, 'CE')
            ce_g  = calculate_greeks(spot, strike, T, r, ce_iv, 'CE')
            # PE Greeks
            pe_iv = calculate_iv(row["pe_ltp"], spot, strike, T, r, 'PE')
            pe_g  = calculate_greeks(spot, strike, T, r, pe_iv, 'PE')

            df.at[idx, "ce_iv"]    = ce_iv
            df.at[idx, "pe_iv"]    = pe_iv
            df.at[idx, "ce_delta"] = ce_g["delta"]
            df.at[idx, "pe_delta"] = pe_g["delta"]
            df.at[idx, "ce_theta"] = ce_g["theta"]
            df.at[idx, "pe_theta"] = pe_g["theta"]
            df.at[idx, "ce_gamma"] = ce_g["gamma"]
            df.at[idx, "pe_gamma"] = pe_g["gamma"]
            df.at[idx, "ce_vega"]  = ce_g["vega"]
            df.at[idx, "pe_vega"]  = pe_g["vega"]
    except Exception as e:
        log.error(f"Greeks calculation error: {e}")
    return df


@retry_api(retries=3, delay=1.0)
def get_live_snapshot(kite: KiteConnect, strikes: list[dict]) -> pd.DataFrame:
    """Fetch real-time quotes including Volume, OI, and calculate Greeks (Zerodha Only)."""
    client = _get_client(kite)
    spot   = get_spot_price(kite)
    expiry = strikes[0]["expiry"] if strikes else ""
    
    # Always fetch from Zerodha Kite for intelligence
    tokens = []
    for s in strikes: 
        tokens.extend([f"NFO:{s['ce_symbol']}", f"NFO:{s['pe_symbol']}"])
        
    quotes = client.quote(tokens)
    rows   = []
    for s in strikes:
        ce_q = quotes.get(f"NFO:{s['ce_symbol']}", {})
        pe_q = quotes.get(f"NFO:{s['pe_symbol']}", {})
        ce_lp, pe_lp = ce_q.get("last_price", 0), pe_q.get("last_price", 0)
        
        rows.append({
            "strike":         s["strike"],
            "ce_symbol":      s["ce_symbol"],
            "pe_symbol":      s["pe_symbol"],
            "ce_ltp":         ce_lp,
            "pe_ltp":         pe_lp,
            "straddle_price": ce_lp + pe_lp,
            "ce_volume":      float(ce_q.get("volume", 0)),
            "pe_volume":      float(pe_q.get("volume", 0)),
            "oi_ce":          float(ce_q.get("oi", 0)),
            "oi_pe":          float(pe_q.get("oi", 0)),
            "atm":            s.get("atm", 0)
        })
        
    df = pd.DataFrame(rows)
    processed_df = _add_greeks_to_df(df, spot, expiry)
    
    # ── Debug: Verify Greeks (Intelligence) ──
    if not processed_df.empty:
        row = processed_df.iloc[len(processed_df)//2] # ATM Row
        log.info(f"Greeks DEBUG (ATM): Strike {row['strike']} | IV {row['ce_iv']:.1%} "
                 f"| Delta {row['ce_delta']:.2f} | Theta {row['ce_theta']:.2f}")
    
    return processed_df