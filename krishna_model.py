# ============================================================
#  ml_model.py — Strike Selector ML Model
# ============================================================
"""
Responsibilities:
  1. Build feature matrix from PostgreSQL straddle_candles + market_context
  2. Label each day: which strike gave MAX P&L
  3. Train XGBoost classifier to predict best strike at 9:20
  4. Expose predict() for live use
  5. Save/load model artifacts
"""

import os
import logging
import datetime as dt
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

import config
import db

os.makedirs(config.MODELS_PATH, exist_ok=True)
os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, "ml_model.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

MODEL_FILE   = os.path.join(config.MODELS_PATH, "krishna_model.pkl")
ENCODER_FILE = os.path.join(config.MODELS_PATH, "label_encoder.pkl")


# ── Feature Engineering ──────────────────────────────────────

def build_features_from_db() -> pd.DataFrame:
    """
    For each (trade_date, strike), extract 9:20 AM snapshot features
    (joined with market_context for VIX/gaps) and simulate P&L.
    Saves rows to daily_features table.
    Returns one row per (trade_date, strike) with features + pnl label.
    """
    with db.get_conn() as conn:
        df = pd.read_sql("""
            SELECT sc.*,
                   mc.vix,
                   mc.nifty_prev_day_change,
                   mc.nifty_open_gap
            FROM straddle_candles sc
            LEFT JOIN market_context mc ON sc.trade_date = mc.trade_date
            ORDER BY sc.trade_date, sc.strike, sc.ts
        """, conn)

    if df.empty:
        log.warning("No candle data found in DB.")
        return pd.DataFrame()

    df["ts"]   = pd.to_datetime(df["ts"])
    df["time"] = df["ts"].dt.time

    results = []
    for (trade_date, strike), grp in df.groupby(["trade_date", "strike"]):
        grp = grp.sort_values("ts").reset_index(drop=True)
        
        # Ensure we have a significant portion of the trading day
        if len(grp) < 300:
            continue

        snap_920 = grp[grp["time"] <= dt.time(9, 20)]
        if snap_920.empty:
            continue
        row_920 = snap_920.iloc[-1]

        straddle_premium = row_920["straddle_price"]
        vwap_at_920      = row_920["vwap"]
        vwap_gap_pct     = row_920["vwap_gap_pct"] if row_920["vwap_gap_pct"] else 0.0
        ce_ltp           = row_920["ce_close"]
        pe_ltp           = row_920["pe_close"]
        ce_pe_ratio      = ce_ltp / pe_ltp if pe_ltp and pe_ltp > 0 else 1.0
        atm              = int(row_920["atm"])
        dist_from_atm    = float(strike) - float(atm)
        dist_pct         = dist_from_atm / float(atm) * 100 if atm else 0

        expiry      = grp.iloc[0]["expiry"]
        expiry_date = pd.to_datetime(expiry).date() if isinstance(expiry, str) else expiry
        trade_dt    = pd.to_datetime(trade_date).date() if isinstance(trade_date, str) else trade_date
        days_to_exp = (expiry_date - trade_dt).days
        day_of_week = trade_dt.weekday()

        vix                = float(row_920.get("vix") or 0)
        nifty_prev_change  = float(row_920.get("nifty_prev_day_change") or 0)
        nifty_open_gap     = float(row_920.get("nifty_open_gap") or 0)
        oi_ce              = int(row_920.get("ce_oi") or 0)
        oi_pe              = int(row_920.get("pe_oi") or 0)
        oi_total           = oi_ce + oi_pe
        oi_imbalance       = (oi_ce - oi_pe) / oi_total if oi_total > 0 else 0

        pnl = simulate_pnl(grp)

        results.append({
            "trade_date":            str(trade_date),
            "expiry":                str(expiry),
            "strike":                int(strike),
            "atm":                   atm,
            "vwap_gap_pct":          round(vwap_gap_pct, 4),
            "straddle_premium":      round(straddle_premium, 2),
            "ce_pe_ratio":           round(ce_pe_ratio, 4),
            "distance_from_atm":     dist_from_atm,
            "distance_from_atm_pct": round(dist_pct, 4),
            "oi_ce":                 oi_ce,
            "oi_pe":                 oi_pe,
            "oi_imbalance":          round(oi_imbalance, 4),
            "days_to_expiry":        days_to_exp,
            "day_of_week":           day_of_week,
            "vix":                   round(vix, 4),
            "nifty_prev_day_change": round(nifty_prev_change, 4),
            "nifty_open_gap":        round(nifty_open_gap, 4),
            "ce_iv":                 round(float(row_920.get("ce_iv") or 0), 4),
            "pe_iv":                 round(float(row_920.get("pe_iv") or 0), 4),
            "ce_delta":              round(float(row_920.get("ce_delta") or 0), 4),
            "pe_delta":              round(float(row_920.get("pe_delta") or 0), 4),
            "ce_theta":              round(float(row_920.get("ce_theta") or 0), 4),
            "pe_theta":              round(float(row_920.get("pe_theta") or 0), 4),
            "ce_gamma":              round(float(row_920.get("ce_gamma") or 0), 6),
            "pe_gamma":              round(float(row_920.get("pe_gamma") or 0), 6),
            "ce_vega":               round(float(row_920.get("ce_vega") or 0), 4),
            "pe_vega":               round(float(row_920.get("pe_vega") or 0), 4),
            
            # Relative Greeks (Phase 5 Refinement)
            "delta_spread":          round(float(row_920.get("ce_delta") or 0) + float(row_920.get("pe_delta") or 0), 4),
            "theta_ratio":           round(float(row_920.get("ce_theta") or 0) / float(row_920.get("pe_theta") or 0.001), 4),
            "vega_imbalance":        round(float(row_920.get("ce_vega") or 0) - float(row_920.get("pe_vega") or 0), 4),
            
            "pnl":                   round(float(pnl), 2),
        })

    feat_df = pd.DataFrame(results)
    if feat_df.empty:
        return feat_df

    best = feat_df.groupby("trade_date")["pnl"].idxmax()
    feat_df["is_best"] = False
    feat_df.loc[best.values, "is_best"] = True
    feat_df["best_strike"] = feat_df.groupby("trade_date")["pnl"].transform(
        lambda x: feat_df.loc[x.idxmax(), "strike"]
    )

    # ── Save to daily_features table ──
    _save_daily_features(feat_df)

    log.info(f"Features built: {len(feat_df)} rows, {feat_df['trade_date'].nunique()} days")
    return feat_df


def _save_daily_features(feat_df: pd.DataFrame):
    """Upsert feature rows into daily_features PostgreSQL table."""
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in feat_df.iterrows():
                cur.execute("""
                    INSERT INTO daily_features
                    (trade_date, expiry, strike, atm, vwap_gap_pct, straddle_premium,
                     ce_pe_ratio, distance_from_atm, distance_from_atm_pct,
                     oi_ce, oi_pe, oi_imbalance, days_to_expiry, day_of_week,
                     vix, nifty_prev_day_change, nifty_open_gap,
                     ce_iv, pe_iv, ce_delta, pe_delta, 
                     ce_theta, pe_theta, ce_gamma, pe_gamma, 
                     ce_vega, pe_vega,
                     delta_spread, theta_ratio, vega_imbalance,
                     pnl, is_best, best_strike)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (trade_date, strike) DO UPDATE SET
                        vwap_gap_pct          = EXCLUDED.vwap_gap_pct,
                        straddle_premium      = EXCLUDED.straddle_premium,
                        ce_pe_ratio           = EXCLUDED.ce_pe_ratio,
                        distance_from_atm     = EXCLUDED.distance_from_atm,
                        distance_from_atm_pct = EXCLUDED.distance_from_atm_pct,
                        oi_ce                 = EXCLUDED.oi_ce,
                        oi_pe                 = EXCLUDED.oi_pe,
                        oi_imbalance          = EXCLUDED.oi_imbalance,
                        days_to_expiry        = EXCLUDED.days_to_expiry,
                        day_of_week           = EXCLUDED.day_of_week,
                        vix                   = EXCLUDED.vix,
                        nifty_prev_day_change = EXCLUDED.nifty_prev_day_change,
                        nifty_open_gap        = EXCLUDED.nifty_open_gap,
                        ce_iv                 = EXCLUDED.ce_iv,
                        pe_iv                 = EXCLUDED.pe_iv,
                        ce_delta              = EXCLUDED.ce_delta,
                        pe_delta              = EXCLUDED.pe_delta,
                        ce_theta              = EXCLUDED.ce_theta,
                        pe_theta              = EXCLUDED.pe_theta,
                        ce_gamma              = EXCLUDED.ce_gamma,
                        pe_gamma              = EXCLUDED.pe_gamma,
                        ce_vega               = EXCLUDED.ce_vega,
                        pe_vega               = EXCLUDED.pe_vega,
                        delta_spread          = EXCLUDED.delta_spread,
                        theta_ratio           = EXCLUDED.theta_ratio,
                        vega_imbalance        = EXCLUDED.vega_imbalance,
                        pnl                   = EXCLUDED.pnl,
                        is_best               = EXCLUDED.is_best,
                        best_strike           = EXCLUDED.best_strike
                """, (
                    row["trade_date"], row["expiry"], int(row["strike"]),
                    int(row["atm"]), row["vwap_gap_pct"], row["straddle_premium"],
                    row["ce_pe_ratio"], row["distance_from_atm"], row["distance_from_atm_pct"],
                    int(row["oi_ce"]), int(row["oi_pe"]), row["oi_imbalance"],
                    int(row["days_to_expiry"]), int(row["day_of_week"]),
                    row["vix"], row["nifty_prev_day_change"], row["nifty_open_gap"],
                    row["ce_iv"], row["pe_iv"], row["ce_delta"], row["pe_delta"],
                    row["ce_theta"], row["pe_theta"], row["ce_gamma"], row["pe_gamma"],
                    row["ce_vega"], row["pe_vega"],
                    row["delta_spread"], row["theta_ratio"], row["vega_imbalance"],
                    row["pnl"], bool(row["is_best"]), int(row["best_strike"])
                ))
    log.info(f"Saved {len(feat_df)} rows to daily_features.")


def simulate_pnl(grp: pd.DataFrame,
                 sl_multiplier: float = config.SL_MULTIPLIER,
                 hard_exit_time: str  = config.HARD_EXIT_TIME) -> float:
    """
    Simulate VWAP crossover straddle strategy on a single strike's minute data.
    Sell when straddle drops below VWAP, buy back when it rises above VWAP.
    Returns total P&L in points.
    """
    ts_col = "ts" if "ts" in grp.columns else "datetime"
    grp = grp.sort_values(ts_col).reset_index(drop=True)
    hard_exit = dt.time(*[int(x) for x in hard_exit_time.split(":")])

    total_pnl     = 0.0
    position      = None
    entry_premium = 0.0

    for _, row in grp.iterrows():
        t        = pd.to_datetime(row[ts_col]).time()
        straddle = row["straddle_price"]
        vwap     = row["vwap"]
        if pd.isna(vwap) or vwap == 0:
            continue

        if t >= hard_exit and position is not None:
            total_pnl += entry_premium - straddle
            position = None
            continue

        if position is None:
            if straddle < vwap:
                position      = {"entry_time": t}
                entry_premium = straddle
        else:
            if straddle >= entry_premium * sl_multiplier:
                total_pnl += entry_premium - straddle
                position = None
                continue
            if straddle > vwap:
                total_pnl += entry_premium - straddle
                position = None

    if position is not None:
        total_pnl += entry_premium - grp.iloc[-1]["straddle_price"]

    return total_pnl


# ── Model Training ───────────────────────────────────────────

def train_model(feat_df: pd.DataFrame = None):
    """
    Train XGBoost classifier: given 9:20 features → predict best strike.
    Uses TimeSeriesSplit to avoid look-ahead bias.
    Saves model + label encoder to disk.
    """
    if feat_df is None:
        feat_df = build_features_from_db()

    if feat_df.empty or feat_df["trade_date"].nunique() < 1:
        raise ValueError("Not enough historical data. Run backfill first.")

    # Phase 1: Train on ALL strikes (Binary Classification: Is this the Best Strike?)
    # This allows Krishna to learn exactly what a winning strike's features look like vs a losing strike.
    X = feat_df[config.ML_FEATURES].fillna(0)
    y = feat_df["is_best"].astype(int)

    cv_scores = []
    # 3. Only run cross-validation if we have enough samples
    if len(X) >= 50: 
        try:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X)//4))
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Check if validation set has labels not present in training
                if not set(np.unique(y_val)).issubset(set(np.unique(y_tr))):
                    log.warning(f"Fold {fold+1} skipped: Validation set has labels not in Training.")
                    continue

                m = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="mlogloss", random_state=42
                )
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                cv_scores.append(accuracy_score(y_val, m.predict(X_val)))
                log.info(f"Fold {fold+1} accuracy: {cv_scores[-1]:.3f}")
        except Exception as e:
            log.error(f"Cross-validation failed: {e}")
    else:
        log.info("Not enough data or variability for cross-validation.")

    # ── Final binary model on full data ──
    final_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    final_model.fit(X, y, verbose=False)

    importance = pd.Series(
        final_model.feature_importances_,
        index=config.ML_FEATURES
    ).sort_values(ascending=False)
    log.info(f"Feature Importance:\n{importance.to_string()}")
    print("\n📊 Feature Importance:")
    print(importance.to_string())

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)

    print(f"\n✅ Model saved → {MODEL_FILE}")
    if cv_scores:
        print(f"   Mean CV Accuracy: {np.mean(cv_scores):.1%}")
    return final_model


def load_model():
    """Load saved model and label encoder."""
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model


# ── Live Prediction ──────────────────────────────────────────

def predict_best_strike(snapshot_df: pd.DataFrame,
                        vix: float,
                        nifty_prev_change: float,
                        nifty_open_gap: float,
                        vwap_map: dict,
                        trade_date: dt.date,
                        expiry_date: dt.date) -> dict:
    """
    Given live 9:20 snapshot of all 11 strikes, predict which one to trade.
    Returns dict with best_strike, confidence, all_scores.
    """
    model = load_model()

    rows = []
    for _, s in snapshot_df.iterrows():
        strike         = s["strike"]
        atm            = s["atm"]
        straddle_price = s["straddle_price"]
        vwap           = vwap_map.get(strike, straddle_price)
        vwap_gap_pct   = (straddle_price - vwap) / vwap * 100 if vwap else 0
        ce_pe_ratio    = s["ce_ltp"] / s["pe_ltp"] if s["pe_ltp"] > 0 else 1.0
        oi_ce          = s["oi_ce"]
        oi_pe          = s["oi_pe"]
        oi_total       = oi_ce + oi_pe
        oi_imbalance   = (oi_ce - oi_pe) / oi_total if oi_total > 0 else 0
        dist           = float(strike) - float(atm)
        dist_pct       = dist / float(atm) * 100 if atm else 0
        days_to_exp    = (expiry_date - trade_date).days

        rows.append({
            "strike":                strike,
            "vwap_gap_pct":          vwap_gap_pct,
            "straddle_premium":      straddle_price,
            "ce_pe_ratio":           ce_pe_ratio,
            "distance_from_atm":     dist,
            "distance_from_atm_pct": dist_pct,
            "oi_ce":                 oi_ce,
            "oi_pe":                 oi_pe,
            "oi_imbalance":          oi_imbalance,
            "days_to_expiry":        days_to_exp,
            "day_of_week":           trade_date.weekday(),
            "vix":                   vix,
            "nifty_prev_day_change": nifty_prev_change,
            "nifty_open_gap":        nifty_open_gap,
            "ce_iv":                 float(s.get("ce_iv") or 0),
            "pe_iv":                 float(s.get("pe_iv") or 0),
            "ce_delta":              float(s.get("ce_delta") or 0),
            "pe_delta":              float(s.get("pe_delta") or 0),
            "ce_theta":              float(s.get("ce_theta") or 0),
            "pe_theta":              float(s.get("pe_theta") or 0),
            "ce_gamma":              float(s.get("ce_gamma") or 0),
            "pe_gamma":              float(s.get("pe_gamma") or 0),
            "ce_vega":               float(s.get("ce_vega") or 0),
            "pe_vega":               float(s.get("pe_vega") or 0),
            
            # Phase 5 Refinement
            "delta_spread":          float(s.get("ce_delta") or 0) + float(s.get("pe_delta") or 0),
            "theta_ratio":           float(s.get("ce_theta") or 0) / float(s.get("pe_theta") or 0.001),
            "vega_imbalance":        float(s.get("ce_vega") or 0) - float(s.get("pe_vega") or 0),
        })

    feat_df = pd.DataFrame(rows)
    X = feat_df[config.ML_FEATURES].fillna(0)

    # Predict probability of being the 'Best Strike' (Class 1)
    proba       = model.predict_proba(X)[:, 1]
    strike_list = feat_df["strike"].tolist()

    scores = {s: 0.0 for s in strike_list}
    for idx, strike in enumerate(strike_list):
        scores[strike] = float(proba[idx])

    best_strike = max(scores, key=scores.get)
    confidence  = scores[best_strike]

    log.info(f"Predicted strike: {best_strike} (offset: {best_strike - snapshot_df.iloc[0]['atm']}, confidence: {confidence:.2%})")
    print(f"\n🎯 Predicted Best Strike: {best_strike}  (confidence: {confidence:.1%})")
    for st, sc in sorted(scores.items()):
        marker = " ◀ SELECTED" if st == best_strike else ""
        print(f"   Strike {st}: {sc:.1%}{marker}")

    # Save to DB if requested
    best_features = X.iloc[strike_list.index(best_strike)].to_dict()
    log_prediction_to_db(trade_date, int(best_strike), float(confidence), best_features)

    return {"best_strike": int(best_strike), "confidence": confidence, "all_scores": scores}


def log_prediction_to_db(trade_date, predicted_strike, confidence, features_dict, 
                         actual_best_strike=None, prediction_correct=None):
    """Save daily prediction and features to PostgreSQL for later inspection."""
    import json
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO krishna_predictions
                    (trade_date, predicted_strike, confidence, features_json, 
                     actual_best_strike, prediction_correct)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_date) DO UPDATE SET
                    predicted_strike  = EXCLUDED.predicted_strike,
                    confidence        = EXCLUDED.confidence,
                    features_json     = EXCLUDED.features_json,
                    actual_best_strike = EXCLUDED.actual_best_strike,
                    prediction_correct = EXCLUDED.prediction_correct
                """, (trade_date, int(predicted_strike), float(confidence), 
                      json.dumps(features_dict), actual_best_strike, prediction_correct))
    except Exception as e:
        log.error(f"Failed to log Krishna prediction to DB: {e}")
