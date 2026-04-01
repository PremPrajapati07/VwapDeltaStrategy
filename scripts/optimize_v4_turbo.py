# ============================================================
#  scripts/optimize_v4_turbo.py — Final Turbo Optimization
# ============================================================
import os
import sys
import pandas as pd
import datetime as dt
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent to path
sys.path.append(os.getcwd())

import config
import db
import arjun_model
import krishna_model as ml

def precalculate_and_filter(feat_df, all_candles):
    """
    1. Pre-calculate Krishna's strike choice for every day.
    2. Filter all_candles to ONLY include data for those specific strikes.
    """
    print("🧠 Pre-calculating Krishna's daily strike choices...")
    model_k = ml.load_model()
    days = sorted(feat_df['trade_date'].unique())
    choices = {}
    selected_keys = []
    
    for trade_date in days:
        day_feat = feat_df[feat_df['trade_date'] == trade_date]
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        proba = model_k.predict_proba(X_rows)[:, 1]
        best_idx = proba.argmax()
        strike = int(day_feat.iloc[best_idx]["strike"])
        confidence = float(proba[best_idx])
        
        choices[trade_date] = {"strike": strike, "confidence": confidence}
        selected_keys.append((trade_date, strike))

    print(f"📦 Filtering 3.5M candles down to ~90k relevant rows...")
    # Convert list of tuples to a filter
    filter_df = pd.DataFrame(selected_keys, columns=['trade_date', 'strike'])
    # Fast join-based filtering
    reduced_candles = pd.merge(all_candles, filter_df, on=['trade_date', 'strike'])
    
    return choices, reduced_candles

def backtest_worker(params, k_choices, reduced_candles, arjun_model_obj):
    """Single iteration of a 1-year backtest with pre-filtered data."""
    k_conf_limit, a_thresh, p_target, s_loss = params
    
    total_pnl = 0.0
    win_days = 0
    total_days = 0
    trade_count = 0
    daily_pnls = []
    
    # Sort days to ensure consistency
    days = sorted(k_choices.keys())
    
    for trade_date in days:
        choice = k_choices[trade_date]
        
        # Guard 1: Krishna Confidence
        if choice['confidence'] < k_conf_limit:
            continue
            
        # Get pre-filtered strike candles for this day
        strike_candles = reduced_candles[reduced_candles['trade_date'] == trade_date]
        if strike_candles.empty: continue
        
        # Use the fast simulator
        day_trades = arjun_model.simulate_pnl_with_arjun_v2(
            strike_candles, 
            arjun_model=arjun_model_obj, 
            threshold=a_thresh,
            p_target=p_target,
            s_loss=s_loss
        )
        
        day_pnl = sum(t["pnl_pts"] for t in day_trades)
        total_pnl += day_pnl
        trade_count += len(day_trades)
        total_days += 1
        if day_pnl > 0: win_days += 1
        daily_pnls.append(day_pnl)

    win_rate = win_days / total_days if total_days > 0 else 0
    
    # Max Drawdown
    max_dd = 0.0
    if daily_pnls:
        cp = pd.Series(daily_pnls).cumsum()
        max_dd = (cp.cummax() - cp).max()

    return {
        "k_conf": k_conf_limit, "a_thresh": a_thresh, "p_target": p_target, "s_loss": s_loss,
        "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
        "trade_count": trade_count, "avg_trades": round(trade_count/total_days, 1) if total_days > 0 else 0,
        "days_traded": total_days, "max_dd": round(max_dd, 1)
    }

def run_turbo_optimizer():
    print("\n🚀 TURBO STRATEGY OPTIMIZER (v4) — FULL YEAR 2025")
    start_date = "2025-01-01"
    end_date   = "2025-12-31"

    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        print("📦 Loading 2025 candles...")
        all_candles = pd.read_sql(f"""
            SELECT trade_date, strike, ts, straddle_price, vwap, ce_volume, pe_volume, 
                   ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta 
            FROM straddle_candles 
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ts, strike
        """, conn)

    all_candles = all_candles.rename(columns={"ts": "datetime"})
    arjun_m = arjun_model.load_arjun_model()
    
    # Pre-calculating choice and filtering candles
    k_choices, reduced_candles = precalculate_and_filter(feat_df, all_candles)

    # Grid (108 Combinations)
    grid = {
        "k_conf": [0.35, 0.40, 0.45],           # Thresholds around median 0.46
        "a_thresh": [0.55, 0.60, 0.65], 
        "p_target": [80, 100, 150, 9999],       # 9999 = No Target
        "s_loss": [-75, -100, -150]
    }
    combinations = list(product(*grid.values()))
    
    cores = cpu_count()
    print(f"🔥 Starting grid search across {cores} CPU cores...")
    
    worker_func = partial(backtest_worker, k_choices=k_choices, reduced_candles=reduced_candles, arjun_model_obj=arjun_m)
    
    with Pool(processes=cores) as pool:
        results = pool.map(worker_func, combinations)

    res_df = pd.DataFrame(results)
    # Target: Win Rate > 80% and High Profit
    res_df["score"] = (res_df["total_pnl"] * (res_df["win_rate"]**2)) / (res_df["avg_trades"]**0.5).replace(0, 1)
    res_df = res_df.sort_values("score", ascending=False)
    
    res_path = os.path.join(config.LOGS_PATH, "optimization_v4_turbo_results.csv")
    res_df.to_csv(res_path, index=False)
    
    print("\n✅ TURBO OPTIMIZATION COMPLETE")
    print(res_df.head(10).to_string(index=False))
    print(f"\nFinal Winners Log: {res_path}")

if __name__ == "__main__":
    run_turbo_optimizer()
