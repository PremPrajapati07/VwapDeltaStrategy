# ============================================================
#  krishna_v2_model.py — Dynamic Strike Re-evaluation Model
# ============================================================
"""
Krishna V2 improves on V1 by evaluating strikes CONTINUOUSLY,
not just at the 9:20 AM snapshot. It adds dynamic market features:
  - Spot momentum & drift since open
  - OI velocity (winding/unwinding detection)
  - Volume surge detection
  - Premium decay rate

Optimized with multiprocessing and vectorized pandas for speed.
"""

import os
import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
import pickle
import time as timer
import multiprocessing as mp
from functools import partial

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from zoneinfo import ZoneInfo

import config
import db

IST = ZoneInfo("Asia/Kolkata")

os.makedirs(config.MODELS_PATH, exist_ok=True)
os.makedirs(config.LOGS_PATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, "krishna_v2.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

MODEL_FILE = os.path.join(config.MODELS_PATH, "krishna_v2_model.pkl")

# ── V2 Feature Set ───────────────────────────────────────────
V2_FEATURES = [
    # V1 features (computed at time t)
    "vwap_gap_pct", "straddle_premium", "ce_pe_ratio",
    "distance_from_atm", "distance_from_atm_pct",
    "oi_ce", "oi_pe", "oi_imbalance",
    "days_to_expiry", "day_of_week",
    "vix", "nifty_prev_day_change", "nifty_open_gap",
    "ce_iv", "pe_iv", "ce_delta", "pe_delta",
    "ce_theta", "pe_theta", "ce_gamma", "pe_gamma",
    "ce_vega", "pe_vega",
    "delta_spread", "theta_ratio", "vega_imbalance",
    # NEW: Dynamic Features
    "spot_change_pct",
    "spot_momentum_5m",
    "oi_velocity_ce",
    "oi_velocity_pe",
    "oi_buildup_ratio",
    "premium_decay_rate",
    "volume_surge",
    "minutes_since_open",
]

# Sample times as minutes-since-midnight for fast integer comparison
_SAMPLE_MINUTES = []
for h in range(9, 11):
    for m in range(0, 60, 5):
        mins = h * 60 + m
        if 9 * 60 + 25 <= mins <= 11 * 60:
            _SAMPLE_MINUTES.append(mins)

_T920_MINS = 9 * 60 + 20
_T915_MINS = 9 * 60 + 15


# ── Per-Day Worker (for multiprocessing) ─────────────────────

def _process_one_day(args):
    """
    Process a single day's candle data into V2 feature rows.
    Designed to be called via multiprocessing.Pool.map().
    """
    day_candles_dict, mc, days_to_exp, day_of_week, date_str = args

    # Reconstruct DataFrame from dict (for pickling across processes)
    day_candles = pd.DataFrame(day_candles_dict)

    if len(day_candles) < 200:
        return []

    # Pre-compute minutes-since-midnight column (integer, fast)
    day_candles["_mins"] = day_candles["_hour"] * 60 + day_candles["_minute"]

    strikes = sorted(day_candles["strike"].unique())

    # Build 9:20 baseline for each strike (vectorized)
    baseline_920 = {}
    for strike in strikes:
        sd = day_candles[day_candles["strike"] == strike]
        snap = sd[sd["_mins"] <= _T920_MINS]
        if snap.empty:
            continue
        r = snap.iloc[-1]
        baseline_920[strike] = {
            "spot": float(r["synthetic_spot"]),
            "straddle": float(r["straddle_price"]),
            "ce_oi": float(r["ce_oi"]),
            "pe_oi": float(r["pe_oi"]),
        }

    if not baseline_920:
        return []

    # Pre-group by strike for fast lookup
    strike_groups = {strike: day_candles[day_candles["strike"] == strike] for strike in strikes}

    all_rows = []

    for sample_mins in _SAMPLE_MINUTES:
        forward_pnls = {}

        # Step 1: compute forward P&L for each strike at this sample time
        for strike in strikes:
            sd = strike_groups[strike]
            snap_at_t = sd[sd["_mins"] <= sample_mins]
            if snap_at_t.empty:
                continue

            forward = sd[sd["_mins"] >= sample_mins]
            if len(forward) < 10:
                continue

            # Fast forward P&L simulation
            fwd_pnl = _fast_forward_pnl(forward)
            forward_pnls[strike] = fwd_pnl

        if len(forward_pnls) < 3:
            continue

        best_strike_at_t = max(forward_pnls, key=forward_pnls.get)
        minutes_since_open = sample_mins - _T915_MINS
        minutes_from_920 = max(sample_mins - _T920_MINS, 1)

        # Step 2: build feature row for each strike
        for strike in forward_pnls:
            sd = strike_groups[strike]
            snap_at_t = sd[sd["_mins"] <= sample_mins]
            if snap_at_t.empty:
                continue
            row_t = snap_at_t.iloc[-1]

            base = baseline_920.get(strike)
            if not base:
                continue

            # V1 features at time t
            straddle_t = float(row_t["straddle_price"])
            vwap_t = float(row_t["vwap"]) if float(row_t["vwap"]) > 0 else straddle_t
            ce_close = float(row_t["ce_close"])
            pe_close = float(row_t["pe_close"])
            atm_val = int(row_t["atm"])
            oi_ce = float(row_t["ce_oi"])
            oi_pe = float(row_t["pe_oi"])
            oi_total = oi_ce + oi_pe

            # Dynamic features
            spot_t = float(row_t["synthetic_spot"])
            spot_920 = base["spot"]
            spot_change_pct = ((spot_t - spot_920) / spot_920 * 100) if spot_920 > 0 else 0

            # Spot momentum (5 min lookback)
            snap_5m = sd[sd["_mins"] <= (sample_mins - 5)]
            if not snap_5m.empty:
                spot_5m = float(snap_5m.iloc[-1]["synthetic_spot"])
                spot_momentum_5m = ((spot_t - spot_5m) / spot_5m * 100) if spot_5m > 0 else 0
            else:
                spot_momentum_5m = 0

            oi_ce_920 = base["ce_oi"]
            oi_pe_920 = base["pe_oi"]
            oi_vel_ce = ((oi_ce - oi_ce_920) / oi_ce_920 * 100) if oi_ce_920 > 0 else 0
            oi_vel_pe = ((oi_pe - oi_pe_920) / oi_pe_920 * 100) if oi_pe_920 > 0 else 0
            oi_buildup = (oi_vel_ce / oi_vel_pe) if abs(oi_vel_pe) > 0.01 else 0

            premium_decay = (base["straddle"] - straddle_t) / minutes_from_920

            total_vol = float(row_t["ce_volume"]) + float(row_t["pe_volume"])
            vol_data = sd[sd["_mins"] <= sample_mins]
            if len(vol_data) > 1:
                avg_vol = (vol_data["ce_volume"].astype(float) + vol_data["pe_volume"].astype(float)).mean()
                volume_surge = (total_vol / avg_vol) if avg_vol > 0 else 1.0
            else:
                volume_surge = 1.0

            ce_theta_v = float(row_t.get("ce_theta") or 0)
            pe_theta_v = float(row_t.get("pe_theta") or 0)

            all_rows.append({
                "trade_date": date_str,
                "strike": int(strike),
                "sample_mins": sample_mins,
                # V1
                "vwap_gap_pct": round((straddle_t - vwap_t) / vwap_t * 100, 4) if vwap_t else 0,
                "straddle_premium": round(straddle_t, 2),
                "ce_pe_ratio": round(ce_close / pe_close, 4) if pe_close > 0 else 1.0,
                "distance_from_atm": float(strike) - float(atm_val),
                "distance_from_atm_pct": round((float(strike) - float(atm_val)) / float(atm_val) * 100, 4) if atm_val else 0,
                "oi_ce": oi_ce,
                "oi_pe": oi_pe,
                "oi_imbalance": round((oi_ce - oi_pe) / oi_total, 4) if oi_total > 0 else 0,
                "days_to_expiry": days_to_exp,
                "day_of_week": day_of_week,
                "vix": mc["vix"],
                "nifty_prev_day_change": mc["nifty_prev_day_change"],
                "nifty_open_gap": mc["nifty_open_gap"],
                "ce_iv": round(float(row_t.get("ce_iv") or 0), 4),
                "pe_iv": round(float(row_t.get("pe_iv") or 0), 4),
                "ce_delta": round(float(row_t.get("ce_delta") or 0), 4),
                "pe_delta": round(float(row_t.get("pe_delta") or 0), 4),
                "ce_theta": round(ce_theta_v, 4),
                "pe_theta": round(pe_theta_v, 4),
                "ce_gamma": round(float(row_t.get("ce_gamma") or 0), 6),
                "pe_gamma": round(float(row_t.get("pe_gamma") or 0), 6),
                "ce_vega": round(float(row_t.get("ce_vega") or 0), 4),
                "pe_vega": round(float(row_t.get("pe_vega") or 0), 4),
                "delta_spread": round(float(row_t.get("ce_delta") or 0) + float(row_t.get("pe_delta") or 0), 4),
                "theta_ratio": round(ce_theta_v / pe_theta_v, 4) if abs(pe_theta_v) > 0.001 else 0,
                "vega_imbalance": round(float(row_t.get("ce_vega") or 0) - float(row_t.get("pe_vega") or 0), 4),
                # V2 dynamic
                "spot_change_pct": round(spot_change_pct, 4),
                "spot_momentum_5m": round(spot_momentum_5m, 4),
                "oi_velocity_ce": round(oi_vel_ce, 4),
                "oi_velocity_pe": round(oi_vel_pe, 4),
                "oi_buildup_ratio": round(min(max(oi_buildup, -10), 10), 4),
                "premium_decay_rate": round(premium_decay, 4),
                "volume_surge": round(min(volume_surge, 50.0), 4),
                "minutes_since_open": round(minutes_since_open, 1),
                # Labels
                "forward_pnl": round(forward_pnls[strike], 2),
                "is_best_at_t": strike == best_strike_at_t,
            })

    return all_rows


def _fast_forward_pnl(forward_df: pd.DataFrame) -> float:
    """
    Fast vectorized forward P&L simulation.
    Uses numpy arrays instead of iterrows for speed.
    """
    prices = forward_df["straddle_price"].values.astype(float)
    vwaps = forward_df["vwap"].values.astype(float)
    mins = forward_df["_mins"].values

    threshold_factor = 1.0 - getattr(config, "VWAP_ENTRY_THRESHOLD_PCT", 0.0) / 100.0
    hard_exit_mins = 15 * 60  # 15:00

    position = None
    total_pnl = 0.0

    for i in range(len(prices)):
        price = prices[i]
        vwap = vwaps[i] if vwaps[i] > 0 else price

        if mins[i] >= hard_exit_mins:
            if position is not None:
                total_pnl += position - price
                position = None
            break

        if position is None:
            if price <= vwap * threshold_factor:
                position = price
        else:
            if price > vwap:
                total_pnl += position - price
                position = None

    if position is not None:
        total_pnl += position - prices[-1]

    return total_pnl


# ── Feature Engineering (Parallelized) ───────────────────────

def build_v2_features(start_date=None, end_date=None) -> pd.DataFrame:
    """
    Build dynamic V2 feature matrix using multiprocessing.
    Loads data once, splits by day, processes in parallel.
    """
    print("\n📊 Krishna V2: Building dynamic feature matrix...")
    t0 = timer.time()

    with db.get_conn() as conn:
        q = "SELECT * FROM straddle_candles"
        conds = []
        if start_date: conds.append(f"trade_date >= '{start_date}'")
        if end_date:   conds.append(f"trade_date <= '{end_date}'")
        if conds: q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY trade_date, strike, ts"
        candles = pd.read_sql(q, conn)

        q_mc = "SELECT * FROM market_context"
        if conds: q_mc += " WHERE " + " AND ".join(conds)
        mc_df = pd.read_sql(q_mc, conn)

    print(f"   Data loaded in {timer.time() - t0:.1f}s. Candles: {len(candles):,}")

    if candles.empty:
        return pd.DataFrame()

    # Timezone & type conversion
    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"] = candles["ts"].dt.tz_convert(IST)

    # Pre-compute hour/minute as integers (FAST comparisons, no dt.time)
    candles["_hour"] = candles["ts"].dt.hour
    candles["_minute"] = candles["ts"].dt.minute

    num_cols = [
        "ce_iv", "pe_iv", "ce_delta", "pe_delta", "ce_theta", "pe_theta",
        "ce_gamma", "pe_gamma", "ce_vega", "pe_vega", "synthetic_spot",
        "vwap", "vwap_gap_pct", "straddle_price", "ce_close", "pe_close",
        "ce_oi", "pe_oi", "atm", "ce_volume", "pe_volume"
    ]
    for col in num_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    # Market context lookup
    mc_map = {}
    if not mc_df.empty:
        for _, row in mc_df.iterrows():
            mc_map[str(row["trade_date"])] = {
                "vix": float(row.get("vix") or 0),
                "nifty_prev_day_change": float(row.get("nifty_prev_day_change") or 0),
                "nifty_open_gap": float(row.get("nifty_open_gap") or 0),
            }

    # Prepare per-day arguments for multiprocessing
    days = sorted(candles["trade_date"].unique())
    pool_args = []

    # Columns to keep (minimize pickle size)
    keep_cols = ["trade_date", "strike", "atm", "straddle_price", "vwap",
                 "ce_close", "pe_close", "ce_oi", "pe_oi", "ce_volume", "pe_volume",
                 "synthetic_spot", "ce_iv", "pe_iv", "ce_delta", "pe_delta",
                 "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", "ce_vega", "pe_vega",
                 "vwap_gap_pct", "_hour", "_minute", "expiry"]

    for trade_date in days:
        date_str = str(trade_date)
        day_data = candles[candles["trade_date"] == trade_date][keep_cols]

        mc = mc_map.get(date_str, {"vix": 0, "nifty_prev_day_change": 0, "nifty_open_gap": 0})

        expiry = day_data.iloc[0]["expiry"]
        expiry_date = pd.to_datetime(expiry).date() if isinstance(expiry, str) else expiry
        trade_dt = pd.to_datetime(trade_date).date() if isinstance(trade_date, str) else trade_date
        days_to_exp = (expiry_date - trade_dt).days
        day_of_week = trade_dt.weekday()

        # Convert to dict for pickling
        pool_args.append((day_data.to_dict("list"), mc, days_to_exp, day_of_week, date_str))

    # Multiprocessing
    cores = max(1, os.cpu_count() - 2)
    print(f"   Processing {len(days)} days using {cores} CPU cores...")
    t1 = timer.time()

    with mp.Pool(processes=cores) as pool:
        results = pool.map(_process_one_day, pool_args)

    # Flatten results
    all_rows = []
    for day_rows in results:
        all_rows.extend(day_rows)

    print(f"   ✅ Feature engineering done in {timer.time() - t1:.1f}s")

    feat_df = pd.DataFrame(all_rows)
    if feat_df.empty:
        print("   ❌ No features generated.")
        return feat_df

    print(f"   Total rows: {len(feat_df):,} from {feat_df['trade_date'].nunique()} days")
    print(f"   Positive label rate: {feat_df['is_best_at_t'].mean():.1%}")

    return feat_df


# ── Training ─────────────────────────────────────────────────

def train_v2(feat_df: pd.DataFrame = None, start_date=None, end_date=None):
    """Train XGBoost on V2 dynamic features."""
    if feat_df is None:
        feat_df = build_v2_features(start_date=start_date, end_date=end_date)

    if feat_df.empty or len(feat_df) < 100:
        raise ValueError("Not enough V2 training data.")

    X = feat_df[V2_FEATURES].fillna(0)
    y = feat_df["is_best_at_t"].astype(int)

    pos = y.sum()
    neg = len(y) - pos
    spw = neg / pos if pos > 0 else 1.0

    print(f"\n🏋️ Training Krishna V2")
    print(f"   Samples: {len(X):,} | Features: {len(V2_FEATURES)}")
    print(f"   BEST={pos:,} ({pos/len(y):.1%})  OTHER={neg:,}")
    print(f"   scale_pos_weight = {spw:.2f}")

    cv_scores = []
    if len(X) >= 200:
        tscv = TimeSeriesSplit(n_splits=5)
        print("\n   Cross-Validation...")
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                continue
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="logloss", random_state=42,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            acc = accuracy_score(y_val, m.predict(X_val))
            cv_scores.append(acc)
            print(f"   Fold {fold+1}: Accuracy = {acc:.4f}")
        if cv_scores:
            print(f"   ✅ Mean CV Accuracy: {np.mean(cv_scores):.4f}")

    print("\n   Training final model...")
    final = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42,
    )
    final.fit(X, y, verbose=False)

    importance = pd.Series(final.feature_importances_, index=V2_FEATURES).sort_values(ascending=False)
    print("\n📊 V2 Feature Importance:")
    print(importance.to_string())

    y_pred = final.predict(X)
    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=["OTHER", "BEST"]))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final, f)
    print(f"\n✅ Krishna V2 saved → {MODEL_FILE}")

    return final


# ── Inference ────────────────────────────────────────────────

def load_v2_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"V2 model not found: {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict_best_strike_v2(
    snapshot_df, vwap_map, baseline_920, current_time,
    vix, nifty_prev_change, nifty_open_gap,
    trade_date, expiry_date, model=None,
):
    """Score all 11 strikes at any minute and return the best one."""
    if model is None:
        model = load_v2_model()

    mins_since_open = (dt.datetime.combine(dt.date.today(), current_time) -
                       dt.datetime.combine(dt.date.today(), dt.time(9, 15))).total_seconds() / 60
    days_to_exp = (expiry_date - trade_date).days

    rows = []
    for _, s in snapshot_df.iterrows():
        strike = s["strike"]
        atm = s["atm"]
        sp = float(s["straddle_price"])
        vwap = vwap_map.get(strike, sp)
        ce = float(s.get("ce_ltp") or s.get("ce_close") or 0)
        pe = float(s.get("pe_ltp") or s.get("pe_close") or 0)
        oi_ce = float(s.get("oi_ce") or s.get("ce_oi") or 0)
        oi_pe = float(s.get("oi_pe") or s.get("pe_oi") or 0)
        oi_tot = oi_ce + oi_pe
        base = baseline_920.get(strike, {})
        spot_t = float(s.get("synthetic_spot") or s.get("spot") or 0)
        spot_920 = base.get("spot", spot_t)
        straddle_920 = base.get("straddle", sp)
        oi_ce_920 = base.get("ce_oi", oi_ce)
        oi_pe_920 = base.get("pe_oi", oi_pe)
        mins_from_920 = max(mins_since_open - 5, 1)

        oi_vel_ce = ((oi_ce - oi_ce_920) / oi_ce_920 * 100) if oi_ce_920 > 0 else 0
        oi_vel_pe = ((oi_pe - oi_pe_920) / oi_pe_920 * 100) if oi_pe_920 > 0 else 0
        ce_theta_v = float(s.get("ce_theta") or 0)
        pe_theta_v = float(s.get("pe_theta") or 0)

        rows.append({
            "strike": strike,
            "vwap_gap_pct": (sp - vwap) / vwap * 100 if vwap else 0,
            "straddle_premium": sp,
            "ce_pe_ratio": ce / pe if pe > 0 else 1.0,
            "distance_from_atm": float(strike) - float(atm),
            "distance_from_atm_pct": (float(strike) - float(atm)) / float(atm) * 100 if atm else 0,
            "oi_ce": oi_ce, "oi_pe": oi_pe,
            "oi_imbalance": (oi_ce - oi_pe) / oi_tot if oi_tot > 0 else 0,
            "days_to_expiry": days_to_exp, "day_of_week": trade_date.weekday(),
            "vix": vix, "nifty_prev_day_change": nifty_prev_change, "nifty_open_gap": nifty_open_gap,
            "ce_iv": float(s.get("ce_iv") or 0), "pe_iv": float(s.get("pe_iv") or 0),
            "ce_delta": float(s.get("ce_delta") or 0), "pe_delta": float(s.get("pe_delta") or 0),
            "ce_theta": ce_theta_v, "pe_theta": pe_theta_v,
            "ce_gamma": float(s.get("ce_gamma") or 0), "pe_gamma": float(s.get("pe_gamma") or 0),
            "ce_vega": float(s.get("ce_vega") or 0), "pe_vega": float(s.get("pe_vega") or 0),
            "delta_spread": float(s.get("ce_delta") or 0) + float(s.get("pe_delta") or 0),
            "theta_ratio": ce_theta_v / pe_theta_v if abs(pe_theta_v) > 0.001 else 0,
            "vega_imbalance": float(s.get("ce_vega") or 0) - float(s.get("pe_vega") or 0),
            "spot_change_pct": ((spot_t - spot_920) / spot_920 * 100) if spot_920 > 0 else 0,
            "spot_momentum_5m": 0,
            "oi_velocity_ce": oi_vel_ce, "oi_velocity_pe": oi_vel_pe,
            "oi_buildup_ratio": min(max((oi_vel_ce / oi_vel_pe) if abs(oi_vel_pe) > 0.01 else 0, -10), 10),
            "premium_decay_rate": (straddle_920 - sp) / mins_from_920,
            "volume_surge": 1.0,
            "minutes_since_open": mins_since_open,
        })

    feat_df = pd.DataFrame(rows)
    X = feat_df[V2_FEATURES].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    scores = {int(feat_df.iloc[i]["strike"]): float(proba[i]) for i in range(len(proba))}
    best = max(scores, key=scores.get)

    log.info(f"V2: {best} (conf: {scores[best]:.2%}) at {current_time}")
    return {"best_strike": best, "confidence": scores[best], "all_scores": scores}


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Krishna V2")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    print("\n🔱 Training Krishna V2 (Dynamic Strike Model)\n")
    feat_df = build_v2_features(start_date=args.start, end_date=args.end)
    if not feat_df.empty:
        train_v2(feat_df)
    else:
        print("❌ No features generated.")
