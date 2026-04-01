# ============================================================
#  scripts/optimize_v2.py — Full-Year Parameter Optimization
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
import arjun_model
import krishna_model as ml

def run_optimizer_v2():
    print("\n🚀 FINAL STRATEGY OPTIMIZER (v2) STARTING — FULL YEAR 2025")
    
    # 1. Define Parameter Grid (144 Combinations)
    grid = {
        "k_conf": [0.0, 0.35, 0.40, 0.45],       # Confidence filters
        "a_thresh": [0.55, 0.60, 0.65],          # Arjun Exits
        "p_target": [80, 100, 120, 9999],        # Profit Targets
        "s_loss": [-75, -100, -150]              # Stop Losses
    }
    
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    print(f"🔬 Total Iterations: {len(combinations)}")
    
    # 2. Cache All 2025 Data
    start_date = "2025-01-01"
    end_date   = "2025-12-31"
    
    print("📦 Loading 2025 features and candles (this may take a minute)...")
    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        all_candles = pd.read_sql(f"""
            SELECT trade_date, strike, ts, straddle_price, vwap, ce_volume, pe_volume, 
                   ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta 
            FROM straddle_candles 
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ts, strike
        """, conn)
    
    all_candles = all_candles.rename(columns={"ts": "datetime"})
    arjun = arjun_model.load_arjun_model()
    days = sorted(feat_df["trade_date"].unique())
    
    results = []

    # 3. Optimization Sweep
    for i, (k_conf, a_thresh, p_target, s_loss) in enumerate(tqdm(combinations, desc="Iterating")):
        # Mock Config
        config.KRISHNA_MIN_CONFIDENCE = k_conf
        config.ARJUN_EXIT_THRESHOLD   = a_thresh
        config.DAILY_PROFIT_TARGET    = p_target
        config.DAILY_STOP_LOSS        = s_loss

        total_pnl = 0.0
        win_days = 0
        total_days = 0
        trade_count = 0
        max_dd = 0.0
        daily_pnls = []
        
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
            daily_pnls.append(day_pnl)

        win_rate = win_days / total_days if total_days > 0 else 0
        
        # Calculate Max DD (Daily Close based)
        if daily_pnls:
            cum_pnl = pd.Series(daily_pnls).cumsum()
            max_dd = (cum_pnl.cummax() - cum_pnl).max()

        results.append({
            "k_conf": k_conf, "a_thresh": a_thresh, "p_target": p_target, "s_loss": s_loss,
            "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
            "trade_count": trade_count, "days_traded": total_days, "max_dd": round(max_dd, 1)
        })

    # 4. Save and Present Top 10
    res_df = pd.DataFrame(results)
    # Filter for Win Rate > 80% as requested
    filtered_df = res_df[res_df["win_rate"] >= 0.8].copy()
    
    if filtered_df.empty:
        print("\n⚠️ No configurations found with Win Rate >= 80%. Broadening to 75%...")
        filtered_df = res_df[res_df["win_rate"] >= 0.75].copy()

    # Score by Profit vs Risk vs Effort
    filtered_df["score"] = (filtered_df["total_pnl"] * (filtered_df["win_rate"]**2)) / (filtered_df["trade_count"]**0.25).replace(0, 1)
    filtered_df = filtered_df.sort_values("score", ascending=False)
    
    res_path = os.path.join(config.LOGS_PATH, "optimization_v2_results.csv")
    filtered_df.to_csv(res_path, index=False)
    
    print("\n✅ OPTIMIZATION COMPLETE — FULL YEAR 2025 Comparison:")
    # Re-order columns for clarity
    cols = ["k_conf", "a_thresh", "p_target", "s_loss", "total_pnl", "win_rate", "trade_count", "days_traded", "score"]
    print(filtered_df[cols].head(10).to_string(index=False))
    print(f"\nReport saved → {res_path}")

if __name__ == "__main__":
    run_optimizer_v2()
