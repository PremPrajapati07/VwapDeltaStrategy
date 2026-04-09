"""
Phase 1: Adaptive Dataset Preparation (Multiprocessing Edition)
===============================================================
Same correctness guarantees as the sequential version — NO look-ahead bias.
Each day is fully isolated (session P&L/strike memory resets daily),
so we can parallelize safely across all CPU cores.

Key Feature: Each "decision row" includes adaptive session-state features:
    session_pnl_so_far      — how the day is going so far
    strike_last_pnl         — result of last trade on this specific strike
    strike_consec_losses    — how many times this strike has lost in a row today

Output: logs/adaptive_training_data.csv
"""

import os
import sys
import datetime as dt
import time as timer
import pickle

import pandas as pd
import numpy as np
import multiprocessing as mp
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db

IST          = ZoneInfo("Asia/Kolkata")
START_DATE   = "2025-01-01"
END_DATE     = "2025-12-31"
SL_MULT      = getattr(config, "SL_MULTIPLIER", 1.5)
HARD_EXIT_M  = 15 * 60        # 15:00 IST in minutes
ENTRY_MINS   = 9 * 60 + 25    # 9:25 AM
SCAN_MINS    = 9 * 60 + 20    # 9:20 AM (baseline snapshot)
THRESHOLD    = getattr(config, "VWAP_ENTRY_THRESHOLD_PCT", 0.2)
COOLDOWN_M   = getattr(config, "SL_COOLDOWN_MINUTES", 15)
MAX_TRADES   = getattr(config, "MAX_TRADES_PER_DAY", 3)
DAILY_SL     = getattr(config, "DAILY_STOP_LOSS", -50)


# ─────────────────────────────────────────────────────────────
# Pure helpers (no DB, no model imports — safe for pickling)
# ─────────────────────────────────────────────────────────────

def _simulate_trade(future_rows_arr, entry_px_idx=0):
    """
    Simulate one trade from a numpy record array of (straddle_price, vwap, _mins).
    Returns (pnl_pts, exit_reason, exit_mins).
    All data is from the current minute onwards → zero look-ahead.
    """
    entry_px = future_rows_arr[entry_px_idx, 0]   # straddle_price
    sl_px    = entry_px * SL_MULT
    entry_t  = int(future_rows_arr[entry_px_idx, 2])  # _mins

    for i in range(entry_px_idx, len(future_rows_arr)):
        px      = future_rows_arr[i, 0]
        vwap    = future_rows_arr[i, 1] if future_rows_arr[i, 1] > 0 else px
        mins_t  = int(future_rows_arr[i, 2])

        if mins_t >= HARD_EXIT_M:
            return entry_px - px, "HARD_EXIT", mins_t
        if px >= sl_px:
            return entry_px - px, "SL", mins_t
        if px > vwap and mins_t > entry_t:
            return entry_px - px, "VWAP_CROSS", mins_t

    last_px = future_rows_arr[-1, 0]
    return entry_px - last_px, "EOD", HARD_EXIT_M


def _build_feature_row(r, vwap, straddle_920, oi_ce_920, oi_pe_920, spot_920,
                        strike, atm_val, ce, pe, oi_ce, oi_pe, spot_t,
                        vix, n_prev_chg, n_open_gap, days_to_exp, day_of_week,
                        session_pnl, trades_taken, last_pnl, consec_losses,
                        sample_mins):
    """Build one feature dict for (strike, time_T). No future data used."""
    sp       = float(r["straddle_price"])
    oi_tot   = oi_ce + oi_pe
    mins_920 = max(sample_mins - SCAN_MINS, 1)

    oi_vel_ce  = ((oi_ce - oi_ce_920) / oi_ce_920 * 100) if oi_ce_920 > 0 else 0
    oi_vel_pe  = ((oi_pe - oi_pe_920) / oi_pe_920 * 100) if oi_pe_920 > 0 else 0
    ce_theta_v = float(r.get("ce_theta") or 0)
    pe_theta_v = float(r.get("pe_theta") or 0)
    spot_chg   = ((spot_t - spot_920) / spot_920 * 100) if spot_920 > 0 else 0

    return {
        # Identifiers
        "strike":                   int(strike),
        "sample_mins":              sample_mins,
        # V2 inherited features
        "vwap_gap_pct":             round((sp - vwap) / vwap * 100, 4) if vwap else 0,
        "straddle_premium":         round(sp, 2),
        "ce_pe_ratio":              round(ce / pe, 4) if pe > 0 else 1.0,
        "distance_from_atm":        float(strike) - atm_val,
        "distance_from_atm_pct":    round((float(strike) - atm_val) / atm_val * 100, 4) if atm_val else 0,
        "oi_ce":                    oi_ce,
        "oi_pe":                    oi_pe,
        "oi_imbalance":             round((oi_ce - oi_pe) / oi_tot, 4) if oi_tot > 0 else 0,
        "days_to_expiry":           days_to_exp,
        "day_of_week":              day_of_week,
        "vix":                      vix,
        "nifty_prev_day_change":    n_prev_chg,
        "nifty_open_gap":           n_open_gap,
        "ce_iv":                    round(float(r.get("ce_iv") or 0), 4),
        "pe_iv":                    round(float(r.get("pe_iv") or 0), 4),
        "ce_delta":                 round(float(r.get("ce_delta") or 0), 4),
        "pe_delta":                 round(float(r.get("pe_delta") or 0), 4),
        "ce_theta":                 round(ce_theta_v, 4),
        "pe_theta":                 round(pe_theta_v, 4),
        "ce_gamma":                 round(float(r.get("ce_gamma") or 0), 6),
        "pe_gamma":                 round(float(r.get("pe_gamma") or 0), 6),
        "ce_vega":                  round(float(r.get("ce_vega") or 0), 4),
        "pe_vega":                  round(float(r.get("pe_vega") or 0), 4),
        "delta_spread":             round(float(r.get("ce_delta") or 0) + float(r.get("pe_delta") or 0), 4),
        "theta_ratio":              round(ce_theta_v / pe_theta_v, 4) if abs(pe_theta_v) > 0.001 else 0,
        "vega_imbalance":           round(float(r.get("ce_vega") or 0) - float(r.get("pe_vega") or 0), 4),
        "spot_change_pct":          round(spot_chg, 4),
        "oi_velocity_ce":           round(oi_vel_ce, 4),
        "oi_velocity_pe":           round(oi_vel_pe, 4),
        "oi_buildup_ratio":         round(min(max((oi_vel_ce / oi_vel_pe) if abs(oi_vel_pe) > 0.01 else 0, -10), 10), 4),
        "premium_decay_rate":       round((straddle_920 - sp) / mins_920, 4),
        "volume_surge":             1.0,
        "minutes_since_open":       round(sample_mins - (9 * 60 + 15), 1),
        # ── Adaptive Session Features (the NEW intelligence) ────
        "session_pnl_so_far":       round(session_pnl, 2),
        "trades_taken_today":       trades_taken,
        "strike_last_pnl":          round(last_pnl, 2),
        "strike_consec_losses":     int(consec_losses),
        "session_is_positive":      int(session_pnl > 0),
        "strike_is_virgin":         int(last_pnl == 0.0 and consec_losses == 0),
    }


# ─────────────────────────────────────────────────────────────
# Worker — processes ONE day
# ─────────────────────────────────────────────────────────────

def _worker_one_day(args):
    """
    Runs a sequential replay for a single trading day.
    Krishna V2 model is loaded from pkl INSIDE the worker (pickling-safe).
    All session state is LOCAL — no global mutation.
    """
    date_str, day_dict, vix, n_prev_chg, n_open_gap, model_file = args

    # Lazy import inside worker (avoids pickling the model object itself)
    import pickle, sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import krishna_v2_model as v2

    trade_date = dt.date.fromisoformat(date_str)
    expiry_str = day_dict["expiry"][0]
    expiry     = pd.to_datetime(expiry_str).date()
    days_to_exp = (expiry - trade_date).days
    day_of_week = trade_date.weekday()

    # Rebuild per-day DataFrame from dict (pickle-safe)
    day_df = pd.DataFrame(day_dict)

    # Load V2 model
    try:
        with open(model_file, "rb") as f:
            krishna = pickle.load(f)
    except Exception:
        return []

    # Per-strike sorted data + numpy arrays for fast simulation
    strikes = sorted(day_df["strike"].unique())
    strike_dfs  = {}
    strike_arrs = {}   # {strike: np.array([ [price, vwap, mins], ... ])}

    for s in strikes:
        sd = day_df[day_df["strike"] == s].sort_values("_mins")
        strike_dfs[s]  = sd.reset_index(drop=True)
        strike_arrs[s] = sd[["straddle_price", "vwap", "_mins"]].to_numpy(dtype=np.float64)

    # 9:20 AM baseline snapshot (for premium_decay_rate and oi_velocity)
    baseline_920 = {}
    for s in strikes:
        sd   = strike_dfs[s]
        snap = sd[sd["_mins"] <= SCAN_MINS]
        if snap.empty: continue
        r = snap.iloc[-1]
        baseline_920[int(s)] = {
            "spot":     float(r["synthetic_spot"]),
            "straddle": float(r["straddle_price"]),
            "ce_oi":    float(r["ce_oi"]),
            "pe_oi":    float(r["pe_oi"]),
        }
    if not baseline_920:
        return []

    # ── Per-day session state (never shared across days) ─────
    session_pnl          = 0.0
    trades_taken         = 0
    strike_last_pnl      = {s: 0.0 for s in strikes}   # initialised to 0
    strike_consec_losses = {s: 0   for s in strikes}
    current_mins         = ENTRY_MINS
    all_rows             = []

    while current_mins < HARD_EXIT_M and trades_taken < MAX_TRADES:
        if session_pnl <= DAILY_SL:
            break

        # ── Snapshot at current_mins (no future rows) ─────────
        snap_at_t = {}
        for s in strikes:
            sd   = strike_dfs[s]
            snap = sd[sd["_mins"] <= current_mins]
            if snap.empty: continue
            snap_at_t[s] = snap.iloc[-1]

        if len(snap_at_t) < 3:
            current_mins += 1
            continue

        vwap_map = {s: float(snap_at_t[s]["vwap"]) if float(snap_at_t[s]["vwap"]) > 0
                    else float(snap_at_t[s]["straddle_price"])
                    for s in snap_at_t}

        # ── Forward P&L for LABELING (the model learns this as y) ─
        forward_pnls = {}
        for s in snap_at_t:
            arr   = strike_arrs[s]
            # Find first row at or after current_mins
            idx   = np.searchsorted(arr[:, 2], current_mins)
            if idx >= len(arr) - 4: continue
            pnl, _, _ = _simulate_trade(arr, idx)
            forward_pnls[s] = pnl

        if len(forward_pnls) < 3:
            current_mins += 5
            continue

        best_label = max(forward_pnls, key=forward_pnls.get)

        # ── Ask Krishna V2: which strike would it choose NOW? ─
        snap_df = pd.DataFrame([{
            "strike":         int(s),
            "atm":            float(snap_at_t[s].get("atm") or s),
            "straddle_price": float(snap_at_t[s]["straddle_price"]),
            "vwap":           float(snap_at_t[s]["vwap"]),
            "ce_ltp":         float(snap_at_t[s].get("ce_close") or 0),
            "pe_ltp":         float(snap_at_t[s].get("pe_close") or 0),
            "oi_ce":          float(snap_at_t[s].get("ce_oi") or 0),
            "oi_pe":          float(snap_at_t[s].get("pe_oi") or 0),
            "synthetic_spot": float(snap_at_t[s].get("synthetic_spot") or 0),
            "ce_iv":  float(snap_at_t[s].get("ce_iv") or 0),
            "pe_iv":  float(snap_at_t[s].get("pe_iv") or 0),
            "ce_delta": float(snap_at_t[s].get("ce_delta") or 0),
            "pe_delta": float(snap_at_t[s].get("pe_delta") or 0),
            "ce_theta": float(snap_at_t[s].get("ce_theta") or 0),
            "pe_theta": float(snap_at_t[s].get("pe_theta") or 0),
            "ce_gamma": float(snap_at_t[s].get("ce_gamma") or 0),
            "pe_gamma": float(snap_at_t[s].get("pe_gamma") or 0),
            "ce_vega":  float(snap_at_t[s].get("ce_vega") or 0),
            "pe_vega":  float(snap_at_t[s].get("pe_vega") or 0),
        } for s in snap_at_t])

        try:
            v2_res = v2.predict_best_strike_v2(
                snapshot_df=snap_df,
                vwap_map=vwap_map,
                baseline_920={int(s): baseline_920[int(s)] for s in snap_at_t if int(s) in baseline_920},
                current_time=dt.time(current_mins // 60, current_mins % 60),
                vix=vix, nifty_prev_change=n_prev_chg, nifty_open_gap=n_open_gap,
                trade_date=trade_date, expiry_date=expiry, model=krishna
            )
            chosen = int(v2_res["best_strike"])
        except Exception:
            chosen = int(max(forward_pnls, key=forward_pnls.get))

        # ── Build feature rows for ALL strikes (for training) ─
        for s in snap_at_t:
            r      = snap_at_t[s]
            base   = baseline_920.get(int(s), {})
            feat   = _build_feature_row(
                r=r,
                vwap=vwap_map[s],
                straddle_920=base.get("straddle", float(r["straddle_price"])),
                oi_ce_920=base.get("ce_oi", 0),
                oi_pe_920=base.get("pe_oi", 0),
                spot_920=base.get("spot", 0),
                strike=s,
                atm_val=float(r.get("atm") or s),
                ce=float(r.get("ce_close") or 0),
                pe=float(r.get("pe_close") or 0),
                oi_ce=float(r.get("ce_oi") or 0),
                oi_pe=float(r.get("pe_oi") or 0),
                spot_t=float(r.get("synthetic_spot") or 0),
                vix=vix, n_prev_chg=n_prev_chg, n_open_gap=n_open_gap,
                days_to_exp=days_to_exp, day_of_week=day_of_week,
                session_pnl=session_pnl,
                trades_taken=trades_taken,
                last_pnl=strike_last_pnl.get(s, 0.0),
                consec_losses=strike_consec_losses.get(s, 0),
                sample_mins=current_mins
            )
            feat["trade_date"]      = date_str
            feat["is_best_at_t"]    = int(s == best_label)
            feat["is_chosen_by_v2"] = int(s == chosen)
            feat["forward_pnl"]     = round(forward_pnls.get(s, 0.0), 2)
            all_rows.append(feat)

        # ── Take the trade on V2-chosen strike, update session state ──
        arr_chosen = strike_arrs.get(chosen)
        if arr_chosen is not None:
            idx = np.searchsorted(arr_chosen[:, 2], current_mins)
            if idx < len(arr_chosen) - 3:
                pnl, reason, exit_mins = _simulate_trade(arr_chosen, idx)

                session_pnl            += pnl
                trades_taken           += 1
                strike_last_pnl[chosen]  = pnl
                if pnl < 0:
                    strike_consec_losses[chosen] = strike_consec_losses.get(chosen, 0) + 1
                else:
                    strike_consec_losses[chosen] = 0

                cooldown               = COOLDOWN_M if reason == "SL" else 5
                current_mins           = int(exit_mins) + cooldown
                continue

        current_mins += 5

    return all_rows


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    t0 = timer.time()
    print(f"\n🚀 Adaptive Dataset Preparation — Multiprocessing Edition")
    print(f"   Period: {START_DATE} to {END_DATE}")
    print(f"   Entry Threshold: {THRESHOLD}% | SL Mult: {SL_MULT}x | Max Trades/Day: {MAX_TRADES}\n")

    # ── Load ALL data once in the main process ─────────────────
    print("   📥 Loading candles from DB (~2.7M rows) ...")
    t_db = timer.time()
    with db.get_conn() as conn:
        candles = pd.read_sql(
            f"SELECT trade_date, expiry, strike, atm, ts, straddle_price, vwap, "
            f"ce_close, pe_close, ce_oi, pe_oi, synthetic_spot, ce_volume, pe_volume, "
            f"ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta, "
            f"ce_gamma, pe_gamma, ce_vega, pe_vega "
            f"FROM straddle_candles "
            f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}' "
            f"ORDER BY trade_date, strike, ts",
            conn
        )
        mc_df = pd.read_sql(
            f"SELECT trade_date, vix, nifty_prev_day_change, nifty_open_gap "
            f"FROM market_context "
            f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'",
            conn
        )
    print(f"   ✅ Loaded {len(candles):,} candles, {mc_df['trade_date'].nunique()} days in {timer.time()-t_db:.0f}s")

    # ── Preprocess candles (once) ──────────────────────────────
    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"]    = candles["ts"].dt.tz_convert(IST)
    candles["_mins"] = (candles["ts"].dt.hour * 60 + candles["ts"].dt.minute).astype(np.int16)

    num_cols = ["straddle_price","vwap","ce_close","pe_close","ce_oi","pe_oi",
                "synthetic_spot","ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta",
                "ce_gamma","pe_gamma","ce_vega","pe_vega","ce_volume","pe_volume","atm"]
    for col in num_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0.0)

    # ── Market context map ─────────────────────────────────────
    mc_map = {str(r["trade_date"]): {
        "vix":                  float(r.get("vix") or 0),
        "nifty_prev_day_change": float(r.get("nifty_prev_day_change") or 0),
        "nifty_open_gap":        float(r.get("nifty_open_gap") or 0),
    } for _, r in mc_df.iterrows()}

    # ── Verify model file exists ───────────────────────────────
    model_file = os.path.join(config.MODELS_PATH, "krishna_v2_model.pkl")
    if not os.path.exists(model_file):
        print(f"   ❌ Model not found: {model_file}. Train Krishna V2 first.")
        return

    # ── Split data by day → convert to dict for pickling ──────
    days = sorted(candles["trade_date"].unique())
    keep = ["trade_date","expiry","strike","atm","straddle_price","vwap","_mins",
            "ce_close","pe_close","ce_oi","pe_oi","synthetic_spot",
            "ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta",
            "ce_gamma","pe_gamma","ce_vega","pe_vega","ce_volume","pe_volume"]

    pool_args = []
    for d in days:
        date_str = str(d)
        day_data = candles[candles["trade_date"] == d][keep].copy()
        mc       = mc_map.get(date_str, {"vix": 0, "nifty_prev_day_change": 0, "nifty_open_gap": 0})
        pool_args.append((
            date_str,
            day_data.to_dict("list"),    # dict is picklable
            mc["vix"],
            mc["nifty_prev_day_change"],
            mc["nifty_open_gap"],
            model_file,
        ))

    # ── Parallel processing ────────────────────────────────────
    # Use n_cores - 2 to leave headroom; this runs CPU-bound per-day replay
    cores = max(1, os.cpu_count() - 2)
    print(f"\n   🔥 Replaying {len(days)} days using {cores} CPU cores (multiprocessing)...\n")

    t1 = timer.time()
    all_rows = []
    chunk_size = max(1, len(pool_args) // (cores * 4))   # small chunks to avoid one core hoarding

    with mp.Pool(processes=cores) as pool:
        for i, day_rows in enumerate(pool.imap_unordered(_worker_one_day, pool_args, chunksize=chunk_size), 1):
            all_rows.extend(day_rows)
            if i % 20 == 0 or i == len(days):
                elapsed = timer.time() - t1
                pct     = i / len(days) * 100
                eta     = elapsed / i * (len(days) - i)
                print(f"   [{i:3d}/{len(days)}] {pct:.0f}% | Rows: {len(all_rows):,} "
                      f"| Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    sweep_time = timer.time() - t1
    print(f"\n   ✅ Replay complete in {sweep_time:.0f}s ({sweep_time/len(days):.1f}s/day)")

    # ── Save output ────────────────────────────────────────────
    out_df   = pd.DataFrame(all_rows)
    out_path = os.path.join(config.LOGS_PATH, "adaptive_training_data.csv")
    out_df.to_csv(out_path, index=False)

    pos_rate = out_df["is_best_at_t"].mean() if "is_best_at_t" in out_df.columns else 0
    print(f"\n{'='*65}")
    print(f"  ✅ Adaptive Training Dataset Ready!")
    print(f"  Rows:           {len(out_df):,}")
    print(f"  Days:           {out_df['trade_date'].nunique()}")
    print(f"  Positive rate:  {pos_rate:.1%}  (fraction where this strike was best)")
    print(f"  Saved to:       {out_path}")
    print(f"  Total time:     {timer.time()-t0:.0f}s")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    mp.freeze_support()     # needed on Windows
    main()
