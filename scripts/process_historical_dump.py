import os
import sys
import re
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import config
from scripts.black_scholes_utils import calculate_iv, calculate_greeks

# ── Configuration ───────────────────────────────────────────
RAW_DATA_ROOT = "raw_data"
CHUNK_SIZE   = 5000  # Number of rows per bulk insert
RISK_FREE_RATE = 0.07 # 7%
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

def parse_filename(filename):
    """
    Extracts underlying, expiry, strike, and type from filename.
    Format: NIFTY25-03-0622000CE.csv
    """
    # Pattern for hyphenated: NIFTY25-01-0223600PE
    m1 = re.search(r"NIFTY(\d{2})-(\d{2})-(\d{2})(\d+)(CE|PE)", filename)
    if m1:
        yy, mm, dd, strike, otype = m1.groups()
        expiry = dt.date(2000 + int(yy), int(mm), int(dd))
        return expiry, int(strike), otype

    # Pattern for non-hyphenated: NIFTY25013025100CE
    m2 = re.search(r"NIFTY(\d{2})(\d{2})(\d{2})(\d+)(CE|PE)", filename)
    if m2:
        yy, mm, dd, strike, otype = m2.groups()
        expiry = dt.date(2000 + int(yy), int(mm), int(dd))
        return expiry, int(strike), otype
    
    return None, None, None

def get_market_context_map():
    """Fetch all nifty_open prices for ATM baseline."""
    with db.get_conn() as conn:
        df = pd.read_sql("SELECT trade_date, nifty_open, vix FROM market_context", conn)
    # Ensure keys are indexable dates
    return df.set_index("trade_date").to_dict('index')

def import_to_db(df):
    insert_query = """
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
            vwap           = EXCLUDED.vwap,
            ce_delta       = EXCLUDED.ce_delta,
            pe_delta       = EXCLUDED.pe_delta
    """
    from psycopg2.extras import execute_values
    rows = []
    for r in df.to_dict("records"):
        rows.append((
            r["trade_date"], r["expiry"], r["strike"], r["atm"], r["ts"],
            r["open_ce"], r["high_ce"], r["low_ce"], r["close_ce"], r["volume_ce"], r["oi_ce"],
            r["open_pe"], r["high_pe"], r["low_pe"], r["close_pe"], r["volume_pe"], r["oi_pe"],
            round(float(r["straddle_price"]), 2), int(r["straddle_volume"]), 
            round(float(r["vwap"]), 4), round(float(r["vwap_gap_pct"]), 4),
            round(float(r["synthetic_spot"]), 2), round(float(r["ce_iv"]), 4), round(float(r["pe_iv"]), 4),
            round(float(r["ce_delta"]), 4), round(float(r["ce_theta"]), 4), round(float(r["ce_gamma"]), 6), round(float(r["ce_vega"]), 4),
            round(float(r["pe_delta"]), 4), round(float(r["pe_theta"]), 4), round(float(r["pe_gamma"]), 6), round(float(r["pe_vega"]), 4)
        ))
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_query, rows)

def process_pair_task(p, context_map):
    try:
        col_names = ["date_str", "time_str", "open", "high", "low", "close", "volume", "oi"]
        ce_df = pd.read_csv(p["ce"], header=None, names=col_names)
        pe_df = pd.read_csv(p["pe"], header=None, names=col_names)

        ce_df["date_str"] = ce_df["date_str"].astype(str)
        pe_df["date_str"] = pe_df["date_str"].astype(str)
        
        ce_df["ts"] = pd.to_datetime(ce_df["date_str"] + " " + ce_df["time_str"], format="%Y%m%d %H:%M")
        pe_df["ts"] = pd.to_datetime(pe_df["date_str"] + " " + pe_df["time_str"], format="%Y%m%d %H:%M")

        merged = pd.merge(
            ce_df[["ts", "open", "high", "low", "close", "volume", "oi"]],
            pe_df[["ts", "open", "high", "low", "close", "volume", "oi"]],
            on="ts", suffixes=("_ce", "_pe"), how="inner"
        )

        if merged.empty: return 0

        merged["trade_date"] = merged["ts"].dt.date
        merged["expiry"]     = p["expiry"]
        merged["strike"]     = p["strike"]

        # --- REDUNDANCY FILTER ---
        # Ensures duplicate insertions do not occur across different expiry files downloading overlapping history.
        prev_expiry_limit = p["expiry"] - dt.timedelta(days=7)
        merged = merged[(merged["trade_date"] <= p["expiry"]) & (merged["trade_date"] > prev_expiry_limit)]

        # Filter for trade days we have context for
        merged = merged[merged["trade_date"].apply(lambda d: d in context_map)]
        if merged.empty: return 0

        # Setup ATM
        def get_atm(d):
            n_open = context_map.get(d, {}).get("nifty_open", 0)
            return int(round(n_open / 100) * 100) if n_open else 0
        
        merged["atm"] = merged["trade_date"].map(get_atm)

        # Metrics
        merged["straddle_price"]  = merged["close_ce"] + merged["close_pe"]
        merged["straddle_volume"] = merged["volume_ce"] + merged["volume_pe"]
        merged["synthetic_spot"]  = p["strike"] + merged["close_ce"] - merged["close_pe"]

        # VWAP
        def compute_vwap(grp):
            price = grp["straddle_price"]
            vol   = grp["straddle_volume"]
            vol_sum = vol.cumsum()
            grp["vwap"] = (price * vol).cumsum() / vol_sum.replace(0, 1)
            grp["vwap_gap_pct"] = (price - grp["vwap"]) / grp["vwap"].replace(0, 1) * 100
            return grp

        # Using index since group_keys=False behavior might be tricky, it's safer
        merged = merged.groupby("trade_date", group_keys=False).apply(compute_vwap)
        if 'trade_date' not in merged.columns:
            merged = merged.reset_index()
        merged["trade_date"] = merged["ts"].dt.date
        merged["expiry"]     = p["expiry"]
        merged["strike"]     = p["strike"]

        # Greeks
        expiry_dt = dt.datetime.combine(p["expiry"], dt.time(15, 30))
        iv_ce_list, iv_pe_list = [], []
        g_ce_list, g_pe_list = [], []

        for row in merged.to_dict("records"):
            tte = (expiry_dt - row["ts"].to_pydatetime()).total_seconds() / (365 * 24 * 3600)
            tte = max(1e-9, tte)

            v_ce = calculate_iv(row["close_ce"], row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, "CE")
            v_pe = calculate_iv(row["close_pe"], row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, "PE")
            
            iv_ce_list.append(v_ce)
            iv_pe_list.append(v_pe)

            g_ce = calculate_greeks(row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, v_ce, "CE")
            g_pe = calculate_greeks(row["synthetic_spot"], p["strike"], tte, RISK_FREE_RATE, v_pe, "PE")
            
            g_ce_list.append(g_ce)
            g_pe_list.append(g_pe)

        merged["ce_iv"] = iv_ce_list
        merged["pe_iv"] = iv_pe_list
        merged["ce_delta"] = [g['delta'] for g in g_ce_list]
        merged["ce_theta"] = [g['theta'] for g in g_ce_list]
        merged["ce_gamma"] = [g['gamma'] for g in g_ce_list]
        merged["ce_vega"]  = [g['vega'] for g in g_ce_list]
        merged["pe_delta"] = [g['delta'] for g in g_pe_list]
        merged["pe_theta"] = [g['theta'] for g in g_pe_list]
        merged["pe_gamma"] = [g['gamma'] for g in g_pe_list]
        merged["pe_vega"]  = [g['vega'] for g in g_pe_list]

        import_to_db(merged)
        return len(merged)

    except Exception as e:
        print(f"    ❌ Error {p['strike']} {p['expiry']}: {e}")
        return 0


def process_month(month_name, context_map):
    month_path = os.path.join(RAW_DATA_ROOT, month_name)
    if not os.path.exists(month_path):
        print(f"⚠️ Folder not found: {month_path}")
        return 0

    print(f"\n📂 Processing Folder: {month_name}")

    # 1. Scan folder and find Leg Pairs
    files = []
    for root, dirs, filenames in os.walk(month_path):
        for f in filenames:
            if f.endswith(".csv"):
                expiry, strike, otype = parse_filename(f)
                if expiry:
                    files.append({
                        "path": os.path.join(root, f),
                        "expiry": expiry,
                        "strike": strike,
                        "type": otype
                    })

    df_files = pd.DataFrame(files)
    if df_files.empty:
        print(f"  ❌ No valid CSV files found in {month_name}.")
        return 0

    # Group by Expiry and Strike to find pairs
    pairs = []
    for (expiry, strike), grp in df_files.groupby(["expiry", "strike"]):
        ce_path = grp[grp["type"] == "CE"]["path"].values
        pe_path = grp[grp["type"] == "PE"]["path"].values
        if len(ce_path) > 0 and len(pe_path) > 0:
            pairs.append({
                "expiry": expiry,
                "strike": strike,
                "ce": ce_path[0],
                "pe": pe_path[0]
            })

    print(f"  📦 Found {len(pairs)} CE/PE strike pairs.")

    # 2. Process each pair with Multiprocessing
    total_rows = 0
    max_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pair_task, p, context_map): p for p in pairs}
        for future in tqdm(as_completed(futures), total=len(pairs), desc=f"  Ingesting {month_name}"):
            total_rows += future.result()

    return total_rows

def run_master_ingestion():
    print("🚀 STARTING UNIVERSAL HISTORICAL INGESTION (JAN-DEC 2025) ...")
    context_map = get_market_context_map()
    if not context_map:
        print("❌ No market context available.")
        return

    total_overall = 0
    for month in MONTHS:
        rows = process_month(month, context_map)
        total_overall += rows

    print(f"\n✅ MASTER INGESTION COMPLETE. Total rows imported: {total_overall}")

if __name__ == "__main__":
    import multiprocessing
    # Important for Windows when using multiprocessing
    multiprocessing.freeze_support()
    run_master_ingestion()
