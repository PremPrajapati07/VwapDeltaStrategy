# ============================================================
#  scripts/optimize_v10_sniper.py — Sniper Mode (80% WR + Capture)
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
    k_conf, p_target, s_loss, a_thresh = params
    
    total_pnl = 0.0
    win_days = 0
    total_days = 0
    best_possible_pnl = 0.0
    
    for trade_date, choice in K_CHOICES.items():
        # Krishna Filter
        if choice['confidence'] < k_conf:
            continue
            
        strike_candles = DATA_SLICES.get(trade_date)
        if strike_candles is None: continue
        
        # Simulation
        day_trades = arjun_model.simulate_pnl_with_arjun_v2(
            strike_candles, ARJUN_MODEL, 
            threshold=a_thresh, 
            p_target=p_target, 
            s_loss=s_loss
        )
        
        day_pnl = sum(t["pnl_pts"] for t in day_trades)
        total_pnl += day_pnl
        total_days += 1
        if day_pnl > 0: win_days += 1
        
        # Best Possible for this day (from all strikes) is approx stored in choice['max_pnl']
        best_possible_pnl += choice['day_max_possible']

    win_rate = win_days / total_days if total_days > 0 else 0
    capture_rate = (total_pnl / best_possible_pnl) if best_possible_pnl > 0 else 0
    
    return {
        "k_conf": k_conf, "p_target": p_target, "s_loss": s_loss, "a_thresh": a_thresh,
        "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
        "capture_rate": round(capture_rate, 3), "days": total_days
    }

def run_sniper_optimizer():
    print("\n🎯 SNIPER OPTIMIZER (v10) — SEARCHING FOR 80% WIN RATE + HIGHER CAPTURE")
    start_date = "2025-01-01"
    end_date   = "2025-12-31"

    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        all_candles = pd.read_sql(f"SELECT trade_date, strike, ts as datetime, straddle_price, vwap, ce_iv, pe_iv, ce_volume, pe_volume, ce_delta, pe_delta, ce_theta, pe_theta FROM straddle_candles WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)

    model_k = ml.load_model()
    model_a = arjun_model.load_arjun_model()

    print("🍕 Slicing data & calculating best possible outcomes...")
    slices = {}
    choices = {}
    days = sorted(feat_df['trade_date'].unique())
    for trade_date in days:
        day_feat = feat_df[feat_df['trade_date'] == trade_date]
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        proba = model_k.predict_proba(X_rows)[:, 1]
        best_idx = proba.argmax()
        strike = int(day_feat.iloc[best_idx]["strike"])
        
        choices[trade_date] = {
            "strike": strike, 
            "confidence": float(proba[best_idx]),
            "day_max_possible": day_feat["pnl"].max() # The absolute best strike's vanilla PNL
        }
        day_data = all_candles[(all_candles['trade_date'] == trade_date) & (all_candles['strike'] == strike)]
        if not day_data.empty: slices[trade_date] = day_data

    # Sniper Grid
    grid = {
        "k_conf": [0.0, 0.35, 0.45, 0.55],       # Moderate filters
        "p_target": [50, 70, 100],               # Targets to lock in Win Rate
        "s_loss": [-150, -200],                  # Room to stay in trade
        "a_thresh": [0.65, 0.75, 0.85]           # Holding winners longer
    }
    combinations = list(product(*grid.values()))
    
    print(f"🔥 Running Sniper Search ({len(combinations)} iterations)...")
    with Pool(processes=cpu_count(), initializer=worker_init, initargs=(slices, choices, model_a)) as pool:
        results = pool.map(backtest_worker, combinations)

    res_df = pd.DataFrame(results).sort_values("win_rate", ascending=False)
    
    # Filter for 80% Win Rate and show best PNL/Capture
    elite_results = res_df[res_df['win_rate'] >= 0.80].sort_values("capture_rate", ascending=False)
    
    print("\n✅ SNIPER SEARCH COMPLETE")
    if elite_results.empty:
        print("⚠️ No configurations hit 80% Win Rate with these parameters. Showing top Win Rates:")
        print(res_df.head(15).to_string(index=False))
    else:
        print("🏆 TOP CONFIGURATIONS (Win Rate >= 80%):")
        print(elite_results.head(15).to_string(index=False))

if __name__ == "__main__":
    run_sniper_optimizer()
