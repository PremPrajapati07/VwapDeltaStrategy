# ============================================================
#  arjun_model_v3.py — Model Arjun V3: The Break-Even Strategist
# ============================================================
"""
Model Arjun V3 is a specialized backtest version of Arjun.
It includes the "Break-Even Safeguard": 
If P&L reaches +30 pts, it sets a trailing SL at +2 pts.
"""

import os
import logging
import datetime as dt
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

import config
import db

os.makedirs(config.MODELS_PATH, exist_ok=True)
os.makedirs(config.LOGS_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, "arjun_model_v3.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# --- CRITICAL: V3 uses its own PKL file ---
ARJUN_MODEL_FILE = os.path.join(config.MODELS_PATH, "arjun_model_v3.pkl")

ARJUN_FEATURES = [
    "pnl_pts", "max_pnl_so_far", "drawdown", "vwap_gap_pct", 
    "delta_drift", "theta_velocity", "iv_drift", "rel_vol_15m",
]

def load_training_data() -> pd.DataFrame:
    with db.get_conn() as conn:
        df = pd.read_sql("""
            SELECT trade_date, ts, strike,
                   pnl_pts, max_pnl_so_far, drawdown,
                   vwap_gap_pct, delta_drift, theta_velocity, iv_drift,
                   rel_vol_15m, should_exit
            FROM arjun_training_data
            ORDER BY trade_date, ts
        """, conn)
    if df.empty:
        raise ValueError("No training data found.")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ARJUN_FEATURES + ["should_exit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def train_arjun_v3(df: pd.DataFrame = None):
    if df is None:
        df = load_training_data()
    X = df[ARJUN_FEATURES].fillna(0)
    y = df["should_exit"].astype(int)
    label_counts = y.value_counts()
    exit_count = label_counts.get(1, 0)
    hold_count = label_counts.get(0, 0)
    scale_pos_weight = hold_count / exit_count if exit_count > 0 else 1.0
    
    final_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
    )
    final_model.fit(X, y, verbose=False)
    with open(ARJUN_MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"✅ Arjun V3 model saved → {ARJUN_MODEL_FILE}")
    return final_model

def load_arjun_model():
    if not os.path.exists(ARJUN_MODEL_FILE):
        # Fallback to base arjun_model if v3 doesn't exist yet
        BASE_MODEL = os.path.join(config.MODELS_PATH, "arjun_model.pkl")
        if os.path.exists(BASE_MODEL):
            with open(BASE_MODEL, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"No Arjun model found (V3 or Base).")
    with open(ARJUN_MODEL_FILE, "rb") as f:
        return pickle.load(f)

def predict_exit(pnl_pts, max_pnl_so_far, drawdown, vwap_gap_pct, delta_drift, 
                 theta_velocity, iv_drift, rel_vol_15m, model=None, threshold=0.55):
    if model is None:
        model = load_arjun_model()
    features = np.array([[pnl_pts, max_pnl_so_far, drawdown, vwap_gap_pct, 
                          delta_drift, theta_velocity, iv_drift, rel_vol_15m]])
    proba = model.predict_proba(features)[0][1]
    return {"should_exit": bool(proba >= threshold), "confidence": round(float(proba), 4)}

# --- State for Break-Even SL ---
is_safe_mode = False

def simulate_pnl_with_arjun_v3(grp, arjun_model=None, threshold=0.65, p_target=100.0, s_loss=-50.0):
    global is_safe_mode
    import config as cfg
    hard_exit      = dt.time(15, 0)
    trades         = []
    position       = False
    entry_time     = None
    entry_price    = 0.0
    entry_iv       = 0.0
    max_pnl        = 0.0
    vol_window     = []
    cooldown_until = None
    cumulative_pnl = 0.0
    trade_count    = 0
    is_safe_mode   = False # Reset start of day

    for i, row in grp.iterrows():
        t       = row["_t"] if "_t" in row else (row["datetime"].time() if "datetime" in row else row["_ts_ist"].time())
        ts_full = row["_ts_ist"] if "_ts_ist" in row else row["datetime"]
        price   = float(row["straddle_price"])
        vwap    = float(row["vwap"]) if float(row["vwap"]) > 0 else price
        
        if not position:
            if trade_count >= cfg.MAX_TRADES_PER_DAY: break
            if cumulative_pnl <= s_loss: break
            if trade_count >= cfg.MAX_TRADES_FOR_PROFIT_TARGET and cumulative_pnl >= p_target:
                break
        
        if cooldown_until and ts_full < cooldown_until: continue

        if t >= hard_exit:
            if position:
                pnl_pts = entry_price - price
                trades.append({"pnl_pts": pnl_pts, "exit_reason": "HARD_EXIT"})
                cumulative_pnl += pnl_pts
                trade_count += 1
                position = False
            break

        if not position:
            if price <= (vwap * (1.0 - getattr(cfg, "VWAP_ENTRY_THRESHOLD_PCT", 0.0) / 100.0)):
                entry_time  = ts_full
                entry_price = price
                entry_iv    = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
                max_pnl     = 0.0
                vol_window  = []
                position    = True
                is_safe_mode = False # New trade
        else:
            pnl_pts     = entry_price - price
            max_pnl     = max(max_pnl, pnl_pts)
            drawdown    = max_pnl - pnl_pts
            vol         = float(row.get("ce_volume", 0)) + float(row.get("pe_volume", 0))
            vol_window.append(vol)
            if len(vol_window) > 15: vol_window.pop(0)
            avg_vol     = max(sum(vol_window)/len(vol_window), 1.0)
            rel_vol     = min(vol / avg_vol, 100.0)

            curr_iv     = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
            delta_d     = abs(float(row.get("ce_delta", 0)) + float(row.get("pe_delta", 0)))
            theta_v     = float(row.get("ce_theta", 0)) + float(row.get("pe_theta", 0))
            vwap_gap    = (vwap - price) / vwap * 100 if vwap else 0
            iv_d        = curr_iv - entry_iv
            
            # --- Break-Even Trailing SL (+30 pts -> +2 pts SL) (V3 Innovation) ---
            if not is_safe_mode and pnl_pts >= 30.0:
                is_safe_mode = True 

            decision = predict_exit(
                pnl_pts=pnl_pts, max_pnl_so_far=max_pnl, drawdown=drawdown,
                vwap_gap_pct=vwap_gap, delta_drift=delta_d,
                theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                model=arjun_model, threshold=threshold,
            )

            exit_reason = None
            if is_safe_mode and pnl_pts <= 2.0:
                exit_reason = "BE_SL_TRIGGERED"
            elif decision["should_exit"]:
                exit_reason = f"ARJUN_EXIT({decision['confidence']:.0%})"
            elif price >= vwap:
                exit_reason = "VWAP_CROSS"

            if exit_reason:
                trades.append({
                    "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": ts_full, "exit_price": price,
                    "pnl_pts": pnl_pts, "exit_reason": exit_reason
                })
                cumulative_pnl += pnl_pts
                trade_count += 1
                position = False
                is_safe_mode = False # Reset
                cooldown_until = ts_full + pd.Timedelta(minutes=cfg.SL_COOLDOWN_MINUTES)
    return trades
