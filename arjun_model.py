# ============================================================
#  arjun_model.py — Model Arjun: The Mid-Day Exit Strategist
# ============================================================
"""
Model Arjun is the second half of the Dual Hand-off Architecture.
Krishna picks the strike at 9:20 AM. Arjun then monitors the trade
minute-by-minute and decides the optimal time to EXIT.

Responsibilities:
  1. Load sequential arjun_training_data from PostgreSQL.
  2. Train XGBoost binary classifier: given mid-trade features → predict 'should_exit'.
  3. Expose predict_exit() for live use — called once per minute after entry.
  4. Save/load model artifact.
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
    filename=os.path.join(config.LOGS_PATH, "arjun_model.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

ARJUN_MODEL_FILE = os.path.join(config.MODELS_PATH, "arjun_model.pkl")

# Features Model Arjun uses to decide exit
ARJUN_FEATURES = [
    "pnl_pts",           # Current P&L since entry (in points)
    "max_pnl_so_far",    # Peak profit achieved in this trade
    "drawdown",          # Pullback from peak (max_pnl - pnl_pts)
    "vwap_gap_pct",      # Current distance from VWAP (%)
    "delta_drift",       # Abs sum of CE+PE deltas (directional bias)
    "theta_velocity",    # Rate of theta decay (CE+PE theta)
    "iv_drift",          # Change in IV from trade entry
    "rel_vol_15m",       # Relative volume vs 15m rolling avg
]


# ── Data Loading ──────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """Load arjun_training_data from PostgreSQL and return feature matrix."""
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
        raise ValueError("No training data found in arjun_training_data. Run prepare_arjun_data.py first.")

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    for col in ARJUN_FEATURES + ["should_exit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    log.info(f"Loaded {len(df)} rows, {df['trade_date'].nunique()} days.")
    print(f"   📊 Loaded {len(df):,} training rows from {df['trade_date'].nunique()} trading days.")
    return df


# ── Training ──────────────────────────────────────────────────

def train_arjun(df: pd.DataFrame = None):
    """
    Train XGBoost binary classifier to predict 'should_exit'.
    Uses TimeSeriesSplit to prevent look-ahead bias.
    """
    if df is None:
        df = load_training_data()

    X = df[ARJUN_FEATURES].fillna(0)
    y = df["should_exit"].astype(int)

    label_counts = y.value_counts()
    exit_count = label_counts.get(1, 0)
    hold_count = label_counts.get(0, 0)
    scale_pos_weight = hold_count / exit_count if exit_count > 0 else 1.0

    print(f"   Label Distribution: HOLD={hold_count:,} ({hold_count/len(y):.1%})  EXIT={exit_count:,} ({exit_count/len(y):.1%})")
    print(f"   scale_pos_weight = {scale_pos_weight:.2f}")

    # ── Cross-Validation ──
    cv_scores = []
    tscv = TimeSeriesSplit(n_splits=5)
    print("\n   Running Time-Series Cross-Validation ...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            print(f"   Fold {fold+1}: Skipped (single class)")
            continue

        m = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = m.predict(X_val)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        cv_scores.append(auc)
        print(f"   Fold {fold+1}: AUC = {auc:.4f}")

    if cv_scores:
        print(f"\n   ✅ Mean CV AUC: {np.mean(cv_scores):.4f}")

    # ── Final Model (full data) ──
    print("\n   Training final model on full dataset ...")
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_, index=ARJUN_FEATURES
    ).sort_values(ascending=False)
    print("\n📊 Arjun Feature Importance:")
    print(importance.to_string())

    # Evaluate on full data (training accuracy)
    y_pred_full = final_model.predict(X)
    print("\n📋 Classification Report (Training Set):")
    print(classification_report(y, y_pred_full, target_names=["HOLD", "EXIT"]))

    # Save
    with open(ARJUN_MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\n✅ Arjun model saved → {ARJUN_MODEL_FILE}")

    return final_model


# ── Inference ─────────────────────────────────────────────────

def load_arjun_model():
    """Load the saved Arjun model from disk."""
    if not os.path.exists(ARJUN_MODEL_FILE):
        raise FileNotFoundError(f"Arjun model not found: {ARJUN_MODEL_FILE}. Train it first.")
    with open(ARJUN_MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict_exit(
    pnl_pts: float,
    max_pnl_so_far: float,
    drawdown: float,
    vwap_gap_pct: float,
    delta_drift: float,
    theta_velocity: float,
    iv_drift: float,
    rel_vol_15m: float,
    model=None,
    threshold: float = 0.55,
) -> dict:
    """
    Called live, once per minute while in a trade.
    Returns a dict: {'should_exit': bool, 'confidence': float}
    """
    if model is None:
        model = load_arjun_model()

    features = np.array([[
        pnl_pts, max_pnl_so_far, drawdown,
        vwap_gap_pct, delta_drift, theta_velocity, iv_drift, rel_vol_15m
    ]])

    proba = model.predict_proba(features)[0][1]
    decision = proba >= threshold

    return {
        "should_exit": bool(decision),
        "confidence":  round(float(proba), 4),
    }




def simulate_pnl_with_arjun(
    grp: pd.DataFrame,
    arjun_model=None,
    sl_multiplier: float = None,  # Arjun replaces SL — only used as absolute fallback
    hard_exit_time: str = None,
    threshold: float = 0.55,
) -> tuple:
    """
    Simulate P&L for one strike's candle data using Arjun for exits.
    Returns (pnl_pts, exit_reason, exit_time).

    Steps:
    1. Wait for first minute where straddle < VWAP to enter.
    2. Each subsequent minute, compute Arjun features and call predict_exit().
    3. Exit when Arjun says so, or at hard_exit_time.
    """
    import config as cfg
    hard_exit = dt.time(*[int(x) for x in (hard_exit_time or cfg.HARD_EXIT_TIME).split(":")])

    # Convert timestamps to IST if timezone-aware
    ts_col = "ts" if "ts" in grp.columns else "datetime"
    grp = grp.sort_values(ts_col).reset_index(drop=True)
    grp["_ts"] = pd.to_datetime(grp[ts_col])
    if grp["_ts"].dt.tz is not None:
        grp["_ts_ist"] = grp["_ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        grp["_ts_ist"] = grp["_ts"]
    grp["_t"] = grp["_ts_ist"].dt.time

    # Ensure numeric columns
    for col in ["straddle_price", "vwap", "vwap_gap_pct",
                "ce_delta", "pe_delta", "ce_theta", "pe_theta", "ce_iv", "pe_iv", "straddle_volume"]:
        if col in grp.columns:
            grp[col] = pd.to_numeric(grp[col], errors="coerce").fillna(0)

    position      = False
    entry_price   = 0.0
    entry_iv      = 0.0
    max_pnl       = 0.0
    total_pnl     = 0.0
    vol_window    = []
    exit_reason   = "NO_TRADE"
    exit_time     = None

    for i, row in grp.iterrows():
        t      = row["_t"]
        price  = float(row["straddle_price"])
        vwap   = float(row["vwap"]) if float(row["vwap"]) > 0 else price

        # Hard exit
        if t >= hard_exit:
            if position:
                total_pnl  = entry_price - price
                exit_reason = "HARD_EXIT"
                exit_time   = t
            break

        if not position:
            # Entry: price below VWAP
            if price < vwap:
                entry_price = price
                entry_iv    = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
                max_pnl     = 0.0
                vol_window  = []
                position    = True
        else:
            pnl_pts     = entry_price - price
            max_pnl     = max(max_pnl, pnl_pts)
            drawdown    = max_pnl - pnl_pts

            # Rolling volume for rel_vol
            vol         = float(row.get("straddle_volume", 0))
            vol_window.append(vol)
            if len(vol_window) > 15:
                vol_window.pop(0)
            avg_vol    = max(sum(vol_window) / len(vol_window), 1.0)
            rel_vol    = min(vol / avg_vol, 100.0)

            curr_iv    = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
            delta_d    = abs(float(row.get("ce_delta", 0)) + float(row.get("pe_delta", 0)))
            theta_v    = float(row.get("ce_theta", 0)) + float(row.get("pe_theta", 0))
            vwap_gap   = float(row.get("vwap_gap_pct", 0))
            iv_d       = curr_iv - entry_iv

            # Ask Arjun
            decision = predict_exit(
                pnl_pts=pnl_pts, max_pnl_so_far=max_pnl, drawdown=drawdown,
                vwap_gap_pct=vwap_gap, delta_drift=delta_d,
                theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                model=arjun_model, threshold=threshold,
            )
            if decision["should_exit"]:
                total_pnl   = pnl_pts
                exit_reason = f"ARJUN_EXIT({decision['confidence']:.0%})"
                exit_time   = t
                position    = False
                break

    return total_pnl, exit_reason, exit_time


if __name__ == "__main__":
    print("\n🔱 Training Model Arjun (Exit Strategist) ...\n")
    df = load_training_data()
    train_arjun(df)
    print("\n✅ Done. Run main.py --mode backtest to validate dual-model performance.")

