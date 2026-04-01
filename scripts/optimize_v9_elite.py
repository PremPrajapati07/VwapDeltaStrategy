# ============================================================
#  scripts/optimize_v9_elite.py — Hunting the 80% Win Rate
# ============================================================
import os
import sys
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count

# Add parent to path
sys.path.append(os.getcwd())

import config
import db
import arjun_model
import krishna_model as ml

# --- Global Constants for Workers ---
DATA_SLICES = None
K_CHOICES   = None
ARJUN_MODEL = None

def worker_init(slices, choices, model):
    global DATA_SLICES, K_CHOICES, ARJUN_MODEL
    DATA_SLICES = slices
    K_CHOICES   = choices
    ARJUN_MODEL = model

def backtest_worker(params):
    global DATA_SLICES, K_CHOICES, ARJUN_MODEL
    k_conf, p_target, s_loss = params
    
    total_pnl = 0.0
    win_days = 0
    total_days = 0
    trade_count = 0
    
    # Iterate through days
    for trade_date, choice in K_CHOICES.items():
        # EXTREME FILTER
        if choice['confidence'] < k_conf:
            continue
            
        strike_candles = DATA_SLICES.get(trade_date)
        if strike_candles is None: continue
        
        # We only take ONE trade per day for the highest win rate (First trade only)
        # We simulate manually to ensure "one and done"
        day_trades = arjun_model.simulate_pnl_with_arjun_v2(
            strike_candles, ARJUN_MODEL, threshold=0.55, 
            p_target=p_target, s_loss=s_loss
        )
        
        if day_trades:
            # We take ALL trades of the day (Arjun manages them)
            # but we count the DAY result
            day_pnl = sum(t["pnl_pts"] for t in day_trades)
            total_pnl += day_pnl
            trade_count += len(day_trades)
            total_days += 1
            if day_pnl > 0: win_days += 1

    win_rate = win_days / total_days if total_days > 0 else 0
    return {
        "k_conf": k_conf, "p_target": p_target, "s_loss": s_loss,
        "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
        "days_traded": total_days, "trade_count": trade_count
    }

def run_elite_optimizer():
    print("\n🦅 ELITE WIN-RATE SEARCH (v9) — HUNTING 80% win rate")
    start_date = "2025-01-01"
    end_date   = "2025-12-31"

    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        all_candles = pd.read_sql(f"SELECT trade_date, strike, ts as datetime, straddle_price, vwap, ce_iv, pe_iv, ce_volume, pe_volume, ce_delta, pe_delta, ce_theta, pe_theta FROM straddle_candles WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)

    model_k = ml.load_model()
    model_a = arjun_model.load_arjun_model()

    print("🍕 Slicing data for elite search...")
    slices = {}
    choices = {}
    days = sorted(feat_df['trade_date'].unique())
    for trade_date in days:
        day_feat = feat_df[feat_df['trade_date'] == trade_date]
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        proba = model_k.predict_proba(X_rows)[:, 1]
        best_idx = proba.argmax()
        strike = int(day_feat.iloc[best_idx]["strike"])
        choices[trade_date] = {"strike": strike, "confidence": float(proba[best_idx])}
        day_data = all_candles[(all_candles['trade_date'] == trade_date) & (all_candles['strike'] == strike)]
        if not day_data.empty: slices[trade_date] = day_data

    # Elite Grid (Searching for 80% Win Rate)
    grid = {
        "k_conf": [0.60, 0.65, 0.70, 0.75, 0.80], # Stronger filters
        "p_target": [40, 50, 60, 80],           # Smaller targets for easier wins
        "s_loss": [-150, -200, -250]            # Wider stops to survive bad start
    }
    combinations = list(product(*grid.values()))
    
    print(f"🔥 Running Elite Search ({len(combinations)} iterations)...")
    with Pool(processes=cpu_count(), initializer=worker_init, initargs=(slices, choices, model_a)) as pool:
        results = pool.map(backtest_worker, combinations)

    res_df = pd.DataFrame(results).sort_values("win_rate", ascending=False)
    print("\n✅ ELITE SEARCH COMPLETE")
    print(res_df.head(20).to_string(index=False))

if __name__ == "__main__":
    run_elite_optimizer()
