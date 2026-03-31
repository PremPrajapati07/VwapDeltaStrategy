import os
import re
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import config
from scripts.black_scholes_utils import calculate_iv, calculate_greeks

# ── Configuration ───────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.getcwd(), "raw_data", "march")
TARGET_DATE  = dt.date(2025, 3, 4)
EXPIRY_DATE  = dt.date(2025, 3, 6)
NIFTY_OPEN   = 22021.45
ATM          = 22000
STRIKE_MIN   = 21600
STRIKE_MAX   = 22400
RISK_FREE_RATE = 0.07

def process_march6():
    print(f"🚀 Injecting March 6th Data (Strikes {STRIKE_MIN}-{STRIKE_MAX}) ...")
    
    # 1. Find pairs
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    strikes = {}
    for f in files:
        m = re.search(r"NIFTY25-03-06(\d+)(CE|PE)", f)
        if m:
            strike = int(m.group(1))
            otype = m.group(2)
            if STRIKE_MIN <= strike <= STRIKE_MAX:
                if strike not in strikes: strikes[strike] = {}
                strikes[strike][otype] = f

    pairs = []
    for s, legs in strikes.items():
        if "CE" in legs and "PE" in legs:
            pairs.append({"strike": s, "ce": legs["CE"], "pe": legs["PE"]})

    print(f"📦 Found {len(pairs)} pairs to process.")
    
    for p in tqdm(pairs):
        try:
            col_names = ["date_str", "time_str", "open", "high", "low", "close", "volume", "oi"]
            ce_df = pd.read_csv(os.path.join(RAW_DATA_DIR, p["ce"]), header=None, names=col_names)
            pe_df = pd.read_csv(os.path.join(RAW_DATA_DIR, p["pe"]), header=None, names=col_names)
            
            ce_df["ts"] = pd.to_datetime(ce_df["date_str"].astype(str) + " " + ce_df["time_str"], format="%Y%m%d %H:%M")
            pe_df["ts"] = pd.to_datetime(pe_df["date_str"].astype(str) + " " + pe_df["time_str"], format="%Y%m%d %H:%M")
            
            merged = pd.merge(ce_df, pe_df, on="ts", suffixes=("_ce", "_pe"))
            merged["trade_date"] = merged["ts"].dt.date
            merged = merged[merged["trade_date"] == TARGET_DATE]
            
            if merged.empty: continue
            
            # Calculations
            merged["straddle_price"] = merged["close_ce"] + merged["close_pe"]
            merged["straddle_volume"] = merged["volume_ce"] + merged["volume_pe"]
            merged["synthetic_spot"] = p["strike"] + merged["close_ce"] - merged["close_pe"]
            
            # VWAP
            merged = merged.sort_values("ts")
            vol_sum = merged["straddle_volume"].cumsum()
            merged["vwap"] = (merged["straddle_price"] * merged["straddle_volume"]).cumsum() / vol_sum.replace(0, 1)
            merged["vwap_gap_pct"] = (merged["straddle_price"] - merged["vwap"]) / merged["vwap"].replace(0, 1) * 100
            
            # Greeks
            expiry_dt = dt.datetime.combine(EXPIRY_DATE, dt.time(15, 30))
            iv_ce, iv_pe = [], []
            g_ce, g_pe = [], []
            
            for _, row in merged.iterrows():
                tte = (expiry_dt - row["ts"].to_pydatetime()).total_seconds() / (365 * 24 * 3600)
                tte = max(1e-9, tte)
                
                v_ce = calculate_iv(row["close_ce"], row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, "CE")
                v_pe = calculate_iv(row["close_pe"], row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, "PE")
                iv_ce.append(v_ce); iv_pe.append(v_pe)
                
                gc = calculate_greeks(row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, v_ce, "CE")
                gp = calculate_greeks(row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, v_pe, "PE")
                g_ce.append(gc); g_pe.append(gp)
                
            merged["ce_iv"] = iv_ce; merged["pe_iv"] = iv_pe
            merged["ce_delta"] = [g['delta'] for g in g_ce]; merged["pe_delta"] = [g['delta'] for g in g_pe]
            merged["ce_theta"] = [g['theta'] for g in g_ce]; merged["pe_theta"] = [g['theta'] for g in g_pe]
            merged["ce_gamma"] = [g['gamma'] for g in g_ce]; merged["pe_gamma"] = [g['gamma'] for g in g_pe]
            merged["ce_vega"] = [g['vega'] for g in g_ce]; merged["pe_vega"] = [g['vega'] for g in g_pe]
            
            import_to_db(merged, p["strike"])
            
        except Exception as e:
            print(f"Error {p['strike']}: {e}")

def import_to_db(df, strike):
    from psycopg2.extras import execute_values
    query = """
        INSERT INTO straddle_candles (
            trade_date, expiry, strike, atm, ts,
            ce_open, ce_high, ce_low, ce_close, ce_volume, ce_oi,
            pe_open, pe_high, pe_low, pe_close, pe_volume, pe_oi,
            straddle_price, straddle_volume, vwap, vwap_gap_pct,
            synthetic_spot, ce_iv, pe_iv, 
            ce_delta, ce_theta, ce_gamma, ce_vega,
            pe_delta, pe_theta, pe_gamma, pe_vega
        ) VALUES %s
        ON CONFLICT (trade_date, strike, ts) DO UPDATE SET
            straddle_price = EXCLUDED.straddle_price,
            vwap = EXCLUDED.vwap,
            ce_delta = EXCLUDED.ce_delta,
            pe_delta = EXCLUDED.pe_delta
    """
    rows = []
    for _, r in df.iterrows():
        rows.append((
            TARGET_DATE, EXPIRY_DATE, strike, ATM, r["ts"],
            r["open_ce"], r["high_ce"], r["low_ce"], r["close_ce"], r["volume_ce"], r["oi_ce"],
            r["open_pe"], r["high_pe"], r["low_pe"], r["close_pe"], r["volume_pe"], r["oi_pe"],
            round(r["straddle_price"], 2), int(r["straddle_volume"]), round(r["vwap"], 4), round(r["vwap_gap_pct"], 4),
            round(r["synthetic_spot"], 2), round(r["ce_iv"], 4), round(r["pe_iv"], 4),
            round(r["ce_delta"], 4), round(r["ce_theta"], 4), round(r["ce_gamma"], 6), round(r["ce_vega"], 4),
            round(r["pe_delta"], 4), round(r["pe_theta"], 4), round(r["pe_gamma"], 6), round(r["pe_vega"], 4)
        ))
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

if __name__ == "__main__":
    process_march6()
