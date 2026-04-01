# ============================================================
#  scripts/strategy_optimizer.py — Parameter Optimization
# ============================================================
import os
import sys
import pandas as pd
import datetime as dt
from itertools import product
from tqdm import tqdm

# Add parent to path
sys.path.append(os.getcwd())

import config
import db
import main
import arjun_model
import krishna_model as ml

def run_optimizer():
    print("\n🚀 STRATEGY OPTIMIZER STARTING — Q1 2025 (Jan-Mar)")
    
    # 1. Define Parameter Grid
    grid = {
        "krishna_conf": [0.55, 0.60, 0.65, 0.70],
        "arjun_thresh": [0.55, 0.60, 0.65],
        "profit_target": [60, 80, 100],
        "stop_loss": [-50, -75]
    }
    
    # Generate all combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    print(f"🔬 Total Iterations: {len(combinations)}")
    
    # 2. Get Q1 2025 Features
    start_date = "2025-01-01"
    end_date   = "2025-03-31"
    
    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"""
            SELECT * FROM daily_features 
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """, conn)
    
    if feat_df.empty:
        print("❌ Error: No features found for Q1 2025. Ensure backfill/train are complete.")
        return

    # Load All Straddle Candles (to avoid DB calls in loop)
    with db.get_conn() as conn:
        all_candles = pd.read_sql(f"""
            SELECT trade_date, strike, ts, straddle_price, vwap, ce_volume, pe_volume, 
                   ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta 
            FROM straddle_candles 
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ts, strike
        """, conn)
    
    all_candles = all_candles.rename(columns={"ts": "datetime"})
    arjun = arjun_model.load_arjun_model()

    results = []
    
    # 3. Iteration Loop
    for i, (k_conf, a_thresh, p_target, s_loss) in enumerate(tqdm(combinations, desc="Iterating")):
        # Mock Config in-place
        config.KRISHNA_MIN_CONFIDENCE = k_conf
        config.ARJUN_EXIT_THRESHOLD   = a_thresh
        config.DAILY_PROFIT_TARGET    = p_target
        config.DAILY_STOP_LOSS        = s_loss

        # Run Q1 Backtest
        total_pnl = 0.0
        win_days = 0
        total_days = 0
        trade_count = 0
        
        days = sorted(feat_df["trade_date"].unique())
        
        for trade_date in days:
            day_feat = feat_df[feat_df["trade_date"] == trade_date]
            X_rows = day_feat[config.ML_FEATURES].fillna(0)
            proba = ml.load_model().predict_proba(X_rows)[:, 1]
            best_idx = proba.argmax()
            confidence = proba[best_idx]
            selected_strike = int(day_feat.iloc[best_idx]["strike"])

            # Krishna Filter
            if confidence < k_conf:
                continue
            
            # Simulate Day
            day_candles = all_candles[all_candles["trade_date"] == trade_date]
            strike_candles = day_candles[day_candles["strike"] == selected_strike]
            if strike_candles.empty: continue
            
            day_trades = arjun_model.simulate_pnl_with_arjun(
                strike_candles, arjun_model=arjun, threshold=a_thresh
            )
            
            day_pnl = sum(t["pnl_pts"] for t in day_trades)
            total_pnl += day_pnl
            trade_count += len(day_trades)
            total_days += 1
            if day_pnl > 0: win_days += 1

        win_rate = win_days / total_days if total_days > 0 else 0
        results.append({
            "k_conf": k_conf, "a_thresh": a_thresh, "p_target": p_target, "s_loss": s_loss,
            "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
            "trade_count": trade_count, "days_traded": total_days
        })

    # 4. Save and Present
    res_df = pd.DataFrame(results)
    res_df["score"] = (res_df["total_pnl"] * res_df["win_rate"]) / res_df["trade_count"].replace(0, 1)**0.5
    res_df = res_df.sort_values("score", ascending=False)
    
    res_path = os.path.join(config.LOGS_PATH, "optimization_results_q1.csv")
    res_df.to_csv(res_path, index=False)
    
    print("\n✅ OPTIMIZATION COMPLETE — Q1 Comparison:")
    print(res_df.head(10).to_string(index=False))
    print(f"\nReport saved → {res_path}")

if __name__ == "__main__":
    run_optimizer()
