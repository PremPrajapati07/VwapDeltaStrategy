# ============================================================
#  scripts/optimize_v11_onesignal.py — The True Sniper (One Trade)
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
    k_conf, s_loss, a_thresh = params
    
    total_pnl = 0.0
    win_days = 0
    total_days = 0
    
    for trade_date, choice in K_CHOICES.items():
        if choice['confidence'] < k_conf:
            continue
            
        strike_candles = DATA_SLICES.get(trade_date)
        if strike_candles is None: continue
        
        # --- ONE SIGNAL RULE ---
        # We only take the FIRST trade that triggers and exits. 
        # We don't re-enter.
        import datetime as dt
        hard_exit = dt.time(15, 0)
        
        position = False
        trade_pnl = 0
        trade_happened = False
        entry_price = 0
        entry_time = None
        max_pnl = 0
        iv_start = 0
        vol_window = []

        for i, row in strike_candles.iterrows():
            t = row["datetime"].time()
            price = float(row["straddle_price"])
            vwap = float(row["vwap"])
            
            if t >= hard_exit:
                if position:
                    trade_pnl = entry_price - price
                    trade_happened = True
                break

            if not position and not trade_happened:
                if price < vwap:
                    position = True
                    entry_price = price
                    entry_time = row["datetime"]
                    iv_start = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
            elif position:
                pnl = entry_price - price
                max_pnl = max(max_pnl, pnl)
                
                # Arjun Exit check
                curr_iv = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
                delta_d = abs(float(row.get("ce_delta", 0)) + float(row.get("pe_delta", 0)))
                theta_v = float(row.get("ce_theta", 0)) + float(row.get("pe_theta", 0))
                vwap_gap = (vwap - price) / vwap * 100 if vwap else 0
                iv_d = curr_iv - iv_start
                
                vol = float(row.get("ce_volume", 0)) + float(row.get("pe_volume", 0))
                vol_window.append(vol)
                if len(vol_window) > 15: vol_window.pop(0)
                avg_vol = max(sum(vol_window)/len(vol_window), 1.0)
                rel_vol = min(vol / avg_vol, 100.0)

                # Use Arjun Model
                decision = arjun_model.predict_exit(
                    pnl_pts=pnl, max_pnl_so_far=max_pnl, drawdown=(max_pnl - pnl),
                    vwap_gap_pct=vwap_gap, delta_drift=delta_d,
                    theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                    model=ARJUN_MODEL, threshold=a_thresh
                )
                
                if decision["should_exit"] or price >= vwap or pnl <= s_loss:
                    trade_pnl = pnl
                    trade_happened = True
                    position = False
                    break
        
        if trade_happened:
            total_days += 1
            total_pnl += trade_pnl
            if trade_pnl > 0: win_days += 1

    win_rate = win_days / total_days if total_days > 0 else 0
    return {
        "k_conf": k_conf, "s_loss": s_loss, "a_thresh": a_thresh,
        "total_pnl": round(total_pnl, 1), "win_rate": round(win_rate, 3),
        "days": total_days
    }

def run_onesignal_optimizer():
    print("\n🕊️ ONE-SIGNAL sniper SEARCH (v11) — 2025")
    start_date = "2025-01-01"
    end_date   = "2025-12-31"

    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        all_candles = pd.read_sql(f"SELECT trade_date, strike, ts as datetime, straddle_price, vwap, ce_iv, pe_iv, ce_volume, pe_volume, ce_delta, pe_delta, ce_theta, pe_theta FROM straddle_candles WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)

    model_k = ml.load_model()
    model_a = arjun_model.load_arjun_model()

    print("🍕 Slicing data...")
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

    # One-Signal Grid
    grid = {
        "k_conf": [0.0, 0.45, 0.55, 0.65, 0.75], 
        "s_loss": [-50, -100, -200],                  
        "a_thresh": [0.55, 0.65, 0.75]           
    }
    combinations = list(product(*grid.values()))
    
    print(f"🔥 Running One-Signal Search ({len(combinations)} iterations)...")
    with Pool(processes=cpu_count(), initializer=worker_init, initargs=(slices, choices, model_a)) as pool:
        results = pool.map(backtest_worker, combinations)

    res_df = pd.DataFrame(results).sort_values("win_rate", ascending=False)
    print("\n✅ ONE-SIGNAL SEARCH COMPLETE")
    print(res_df.head(20).to_string(index=False))

if __name__ == "__main__":
    run_onesignal_optimizer()
