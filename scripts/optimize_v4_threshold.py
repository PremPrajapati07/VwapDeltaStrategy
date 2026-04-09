"""
scripts/optimize_v4_threshold.py
==================================
Threshold Sweep for Krishna V3 + Arjun V4 (Adaptive AI).
Aligned with backtest_v4.py logic.

Fixes applied vs previous version:
  1. Models loaded fresh per worker (no None-globals bug) — mirrors backtest_v4.py exactly.
  2. Added idx >= len(arr) - 3 guard to skip trades with no future candles.
  3. Added ce_gamma, pe_gamma, ce_vega, pe_vega to SQL query and keep list.
  4. init_worker / G_MODEL_* globals removed (were unused / broken).
  5. exit_reason initialized to "EOD" before trade loop (UnboundLocalError fix).
"""

import os
import sys
import datetime as dt
import time as timer
import numpy as np
import pandas as pd
import multiprocessing as mp
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db

IST          = ZoneInfo("Asia/Kolkata")
START_DATE   = "2025-01-01"
END_DATE     = "2025-12-31"
THRESHOLDS   = [round(i * 0.1, 1) for i in range(16)]   # [0.0, 0.1, ..., 1.5]

# Constants — must stay in sync with backtest_v4.py
SL_MULT      = getattr(config, "SL_MULTIPLIER", 1.5)
HARD_EXIT_M  = 15 * 60
ENTRY_MINS   = 9 * 60 + 25
SCAN_MINS    = 9 * 60 + 20
COOLDOWN_M   = getattr(config, "SL_COOLDOWN_MINUTES", 15)
MAX_TRADES   = getattr(config, "MAX_TRADES_PER_DAY", 3)
DAILY_SL     = getattr(config, "DAILY_STOP_LOSS", -50)
EXIT_THRESH  = getattr(config, "ARJUN_EXIT_THRESHOLD", 0.55)
BE_SL_ON     = 30.0
BE_SL_OFF    = 2.0

MODEL_V3 = os.path.join(config.MODELS_PATH, "krishna_v3_model.pkl")
MODEL_V4 = os.path.join(config.MODELS_PATH, "arjun_model_v4.pkl")


# ─────────────────────────────────────────────────────────────
# Worker — 100% mirror of backtest_v4.py _worker_day logic
def _worker(args):
    """
    100% Logic Mirror of backtest_v4.py's per-day simulation.
    """
    (threshold, date_str, day_df_dict, vix, n_prev, n_gap) = args
    import pickle, sys, os, datetime as dt
    import numpy as np, pandas as pd
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import krishna_v3_model as v3, arjun_model_v4 as v4

    # Load models fresh per worker
    with open(MODEL_V3, "rb") as f: m_v3 = pickle.load(f)
    with open(MODEL_V4, "rb") as f: m_v4 = pickle.load(f)

    trade_date = dt.date.fromisoformat(date_str)
    day_df     = pd.DataFrame(day_df_dict)
    expiry     = pd.to_datetime(day_df.iloc[0]["expiry"]).date()
    
    # Store strikes as ints consistently
    strikes = [int(s) for s in sorted(day_df["strike"].unique())]

    strike_dfs  = {}
    strike_arrs = {}
    for s in strikes:
        sd = day_df[day_df["strike"] == s].sort_values("_mins")
        strike_dfs[s]  = sd.reset_index(drop=True)
        # Optimized array: [straddle_price, vwap, mins]
        strike_arrs[s] = sd[["straddle_price", "vwap", "_mins"]].to_numpy(dtype=np.float64)

    baseline_920 = {}
    for s in strikes:
        sd   = strike_dfs[s]
        snap = sd[sd["_mins"] <= SCAN_MINS]
        if snap.empty: continue
        r = snap.iloc[-1]
        baseline_920[s] = {
            "spot":     float(r["synthetic_spot"]), 
            "straddle": float(r["straddle_price"]),
            "ce_oi":    float(r["ce_oi"]),         
            "pe_oi":    float(r["pe_oi"]),
        }

    if not baseline_920:
        return {"threshold": threshold, "date": date_str, "pnl": 0.0, "trades": 0}

    session_pnl          = 0.0
    trades_taken         = 0
    strike_last_pnl      = {s: 0.0 for s in strikes}
    strike_consec_losses = {s: 0   for s in strikes}
    blacklist            = {} # {strike: blacklist_expiry_mins}
    current_mins         = ENTRY_MINS
    day_trade_count      = 0

    while current_mins < HARD_EXIT_M and trades_taken < MAX_TRADES:
        if session_pnl <= DAILY_SL: break

        # Snapshotpast data check
        snap_at_t = {}
        for s in strikes:
            sd = strike_dfs[s]
            snap = sd[sd["_mins"] <= current_mins]
            if snap.empty: continue
            snap_at_t[s] = snap.iloc[-1]
        
        if len(snap_at_t) < 3:
            current_mins += 1; continue

        vwap_map = {s: float(snap_at_t[s]["vwap"]) if float(snap_at_t[s]["vwap"]) > 0 
                    else float(snap_at_t[s]["straddle_price"]) for s in snap_at_t}
        available = [s for s in snap_at_t if blacklist.get(s, 0) <= current_mins]
        if not available: available = list(snap_at_t.keys())

        snap_df = pd.DataFrame([{
            "strike": int(s), "atm": float(snap_at_t[s].get("atm") or s),
            "straddle_price": float(snap_at_t[s]["straddle_price"]),
            "vwap": float(snap_at_t[s]["vwap"]),
            "ce_ltp": float(snap_at_t[s].get("ce_close") or 0),
            "pe_ltp": float(snap_at_t[s].get("pe_close") or 0),
            "oi_ce":  float(snap_at_t[s].get("ce_oi") or 0),
            "oi_pe":  float(snap_at_t[s].get("pe_oi") or 0),
            "synthetic_spot": float(snap_at_t[s].get("synthetic_spot") or 0),
            **{col: float(snap_at_t[s].get(col) or 0) for col in 
               ["ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta",
                "ce_gamma","pe_gamma","ce_vega","pe_vega"]},
        } for s in available])

        try:
            pred = v3.predict_best_strike_v3(
                snapshot_df=snap_df, vwap_map={s: vwap_map[s] for s in available},
                baseline_920={s: baseline_920[s] for s in available if s in baseline_920},
                current_time=dt.time(current_mins // 60, current_mins % 60),
                vix=vix, nifty_prev_change=n_prev, nifty_open_gap=n_gap,
                trade_date=trade_date, expiry_date=expiry, model=m_v3,
                session_pnl=session_pnl, trades_taken=trades_taken,
                strike_last_pnl={s: strike_last_pnl.get(s, 0.0) for s in available},
                strike_consec_losses={s: strike_consec_losses.get(s, 0) for s in available},
            )
            chosen = int(pred["best_strike"])
        except Exception:
            chosen = available[len(available)//2]

        chosen_row = snap_at_t.get(chosen)
        if chosen_row is None:
            current_mins += 5; continue
        
        entry_px = float(chosen_row["straddle_price"])
        entry_vwap = vwap_map.get(chosen, entry_px)
        if entry_px > entry_vwap * (1.0 - threshold / 100.0):
            current_mins += 1; continue

        # Trade Loop Start
        arr = strike_arrs.get(chosen)
        if arr is None:
            current_mins += 5; continue
        idx = int(np.searchsorted(arr[:, 2], current_mins))
        if idx >= len(arr) - 3:
            current_mins += 5; continue
        
        sl_px        = entry_px * SL_MULT
        max_pnl_pts  = 0.0
        is_safe_mode = False
        entry_iv     = float(chosen_row.get("ce_iv") or 0) + float(chosen_row.get("pe_iv") or 0)
        vol_window   = []
        exit_t       = HARD_EXIT_M
        pnl_pts      = 0.0
        exit_reason  = "EOD"
        sd_chosen    = strike_dfs[chosen]

        for j in range(idx, len(arr)):
            px     = arr[j, 0]
            vwap_j = arr[j, 1] if arr[j, 1] > 0 else px
            mins_j = int(arr[j, 2])
            pnl_pts = entry_px - px
            
            if mins_j >= HARD_EXIT_M: 
                exit_reason = "HARD_EXIT"; exit_t = mins_j; break
            if px >= sl_px: 
                exit_reason = "SL"; exit_t = mins_j; break
            
            # Break-even SL
            if not is_safe_mode and pnl_pts >= BE_SL_ON: 
                is_safe_mode = True
            if is_safe_mode and pnl_pts <= BE_SL_OFF: 
                exit_reason = "BE_SL"; exit_t = mins_j; break

            max_pnl_pts = max(max_pnl_pts, pnl_pts)
            drawdown    = max_pnl_pts - pnl_pts

            row_j = sd_chosen[sd_chosen["_mins"] == mins_j]
            if not row_j.empty:
                r = row_j.iloc[-1]
                vol = float(r.get("ce_volume") or 0) + float(r.get("pe_volume") or 0)
                vol_window.append(vol)
                if len(vol_window) > 15: vol_window.pop(0)
                avg_vol  = max(sum(vol_window)/len(vol_window), 1.0)
                rel_vol  = min(vol/avg_vol, 100.0)
                curr_iv  = float(r.get("ce_iv") or 0) + float(r.get("pe_iv") or 0)
                iv_d     = curr_iv - entry_iv
                delta_d  = abs(float(r.get("ce_delta") or 0) + float(r.get("pe_delta") or 0))
                theta_v  = float(r.get("ce_theta") or 0) + float(r.get("pe_theta") or 0)
                vwap_gap = (vwap_j - px)/vwap_j * 100 if vwap_j else 0

                try:
                    res = v4.predict_exit_v4(
                        pnl_pts=pnl_pts, max_pnl_so_far=max_pnl_pts, drawdown=drawdown,
                        vwap_gap_pct=vwap_gap, delta_drift=delta_d, theta_velocity=theta_v, iv_drift=iv_d,
                        rel_vol_15m=rel_vol, session_pnl=session_pnl, trades_taken=trades_taken,
                        is_safe_mode=is_safe_mode, model=m_v4, threshold=EXIT_THRESH
                    )
                    if res["should_exit"]: 
                        exit_reason = "ARJUN_V4"; exit_t = mins_j; break
                except Exception: 
                    pass
            
            if px > vwap_j: 
                exit_reason = "VWAP_CROSS"; exit_t = mins_j; break

        # Update Session State
        session_pnl += pnl_pts
        trades_taken += 1
        day_trade_count += 1
        strike_last_pnl[chosen] = pnl_pts
        if pnl_pts < 0:
            strike_consec_losses[chosen] = strike_consec_losses.get(chosen, 0) + 1
            if pnl_pts <= -7.0: 
                # Sync with backtest_v4.py: blacklist from ENTRY time
                blacklist[chosen] = current_mins + 60
        else: 
            strike_consec_losses[chosen] = 0
        
        current_mins = exit_t + (COOLDOWN_M if exit_reason == "SL" else 5)

    return {
        "threshold": threshold, 
        "date": date_str, 
        "pnl": round(session_pnl, 2), 
        "trades": day_trade_count
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    t0 = timer.time()
    print(f"\n🚀 V4 Optimizer — Krishna V3 + Arjun V4 (Aligned with backtest_v4.py)")
    print(f"   Period     : {START_DATE} to {END_DATE}")
    print(f"   Thresholds : {THRESHOLDS}\n")

    print("   📥 Loading candles from DB...")
    with db.get_conn() as conn:
        candles = pd.read_sql(
            # FIX 3: added ce_gamma, pe_gamma, ce_vega, pe_vega to match backtest_v4.py
            f"SELECT trade_date, expiry, strike, atm, ts, straddle_price, vwap, "
            f"ce_close, pe_close, ce_oi, pe_oi, synthetic_spot, ce_volume, pe_volume, "
            f"ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta, "
            f"ce_gamma, pe_gamma, ce_vega, pe_vega "
            f"FROM straddle_candles "
            f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'", conn
        )
        mc_df = pd.read_sql(
            f"SELECT * FROM market_context "
            f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'", conn
        )

    print(f"   ✅ Loaded {len(candles):,} candles, {len(mc_df)} days.")

    ts_ser = pd.to_datetime(candles["ts"])
    if ts_ser.dt.tz is None:
        ts_ser = ts_ser.dt.tz_localize("UTC")
    candles["ts"]    = ts_ser.dt.tz_convert(IST)
    candles["_mins"] = (candles["ts"].dt.hour * 60 + candles["ts"].dt.minute).astype(np.int16)

    # FIX 3: keep list now includes gamma/vega columns
    keep = [
        "expiry", "strike", "atm", "straddle_price", "vwap", "_mins",
        "ce_close", "pe_close", "ce_oi", "pe_oi", "synthetic_spot",
        "ce_iv", "pe_iv", "ce_delta", "pe_delta", "ce_theta", "pe_theta",
        "ce_gamma", "pe_gamma", "ce_vega", "pe_vega",
        "ce_volume", "pe_volume",
    ]

    # Numeric coerce — same as backtest_v4.py
    for col in keep:
        if col in candles.columns and col not in ("expiry", "strike", "atm"):
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0.0)

    mc_map = {str(r["trade_date"]): {
        "vix":    float(r.get("vix") or 0),
        "n_prev": float(r.get("nifty_prev_day_change") or 0),
        "n_gap":  float(r.get("nifty_open_gap") or 0),
    } for _, r in mc_df.iterrows()}

    # Verify model files exist before spawning workers
    for f in [MODEL_V3, MODEL_V4]:
        if not os.path.exists(f):
            print(f"❌ Model not found: {f}"); return

    days      = sorted(candles["trade_date"].unique())
    pool_args = []
    for d in days:
        date_str = str(d)
        mc       = mc_map.get(date_str, {"vix": 0, "n_prev": 0, "n_gap": 0})
        day_dict = candles[candles["trade_date"] == d][keep].to_dict("list")
        for thr in THRESHOLDS:
            pool_args.append((thr, date_str, day_dict, mc["vix"], mc["n_prev"], mc["n_gap"]))

    cores = max(1, os.cpu_count() - 2)
    print(f"   🔥 Starting sweep — {len(pool_args)} runs on {cores} cores...\n")

    all_rows = []
    t1       = timer.time()

    # FIX 4: no initializer= / init_worker — models loaded inside _worker like backtest_v4
    with mp.Pool(processes=cores) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_worker, pool_args, chunksize=max(1, len(pool_args) // (cores * 4))), 1
        ):
            all_rows.append(res)
            if i % 50 == 0 or i == len(pool_args):
                elapsed = timer.time() - t1
                eta     = (elapsed / i) * (len(pool_args) - i)
                print(f"   [{i:4d}/{len(pool_args)}] Complete | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    res_df  = pd.DataFrame(all_rows)
    summary = []

    for thr in THRESHOLDS:
        grp  = res_df[res_df["threshold"] == thr].copy()
        pnls = grp["pnl"].tolist()
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p < 0]
        flat = [p for p in pnls if p == 0]

        import numpy as _np
        cum    = _np.cumsum(pnls)
        max_dd = float((_np.array(cum) - _np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0.0

        summary.append({
            "Threshold":    f"{thr:.1f}%",
            "Trading Days": len(pnls),
            "Total P&L":    round(sum(pnls), 1),
            "Win Rate":     f"{len(wins)/len(pnls)*100:.1f}%" if pnls else "0%",
            "W/L/F":        f"{len(wins)}W/{len(loss)}L/{len(flat)}F",
            "Avg Win Day":  f"{_np.mean(wins):+.1f}" if wins else "0",
            "Avg Loss Day": f"{_np.mean(loss):+.1f}" if loss else "0",
            "Max Drawdown": f"{max_dd:.1f}",
            "Avg/Day":      f"{sum(pnls)/len(pnls):+.1f}" if pnls else "0",
            "Total Trades": int(grp["trades"].sum()),
        })

    print(f"\n{'='*90}")
    print(f"   🔱 V4 OPTIMIZATION RESULTS (Aligned with backtest_v4.py)")
    print(f"{'='*90}")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"{'='*90}")
    print(f"\n⏱️  Total Sweep Time: {timer.time()-t0:.0f}s")

    out_csv = os.path.join(config.LOGS_PATH, "optimization_v4_results.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)

    # Also save per-day breakdown for analysis
    out_daily = os.path.join(config.LOGS_PATH, "optimization_v4_daily.csv")
    res_df.sort_values(["threshold", "date"]).to_csv(out_daily, index=False)

    print(f"📊 Summary saved → {out_csv}")
    print(f"📊 Daily P&L    → {out_daily}\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()