"""
scripts/backtest_v4.py
=======================
Realistic day-by-day backtest for Krishna V3 + Arjun V4.

Usage:
    python scripts/backtest_v4.py --threshold 0.2 --start 2025-01-01 --end 2025-12-31

Design (No Look-Ahead / Realistic):
  - Each day is replayed sequentially from 9:25 AM.
  - Strike selected via Krishna V3 (uses only data up to the prediction moment).
  - Session state (SessionTracker) resets every day — same as live trading.
  - Arjun V4 exit decisions include session_pnl, trades_taken, is_safe_mode.
  - Break-Even SL: +30 pts profit → +2 pts trailing SL.
  - Days are processed in PARALLEL (multiprocessing across days).

Output:
  - logs/backtest_v4_daily_YYYY-MM-DD_to_YYYY-MM-DD_{threshold}.csv
  - Console summary table
"""

import os
import sys
import argparse
import datetime as dt
import time as timer
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db

IST          = ZoneInfo("Asia/Kolkata")
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
# Per-day worker (realistic sequential replay)
# ─────────────────────────────────────────────────────────────

def _worker_day(args):
    """
    Replay a full trading day using Krishna V3 + Arjun V4.
    Returns a dict with all trades and day summary.
    All session state is LOCAL — fresh per day as in live trading.
    """
    (date_str, day_dict, vix, n_prev, n_gap, threshold,
     v3_file, v4_file) = args

    import pickle, sys, os, datetime as dt
    import numpy as np, pandas as pd
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import krishna_v3_model as v3, arjun_model_v4 as v4

    with open(v3_file, "rb") as f: m_v3 = pickle.load(f)
    with open(v4_file, "rb") as f: m_v4 = pickle.load(f)

    trade_date = dt.date.fromisoformat(date_str)
    day_df     = pd.DataFrame(day_dict)
    expiry     = pd.to_datetime(day_df.iloc[0]["expiry"]).date()
    strikes    = sorted(day_df["strike"].unique())

    strike_dfs  = {}
    strike_arrs = {}
    for s in strikes:
        sd = day_df[day_df["strike"] == s].sort_values("_mins")
        strike_dfs[s]  = sd.reset_index(drop=True)
        strike_arrs[s] = sd[["straddle_price","vwap","_mins"]].to_numpy(dtype=np.float64)

    baseline_920 = {}
    for s in strikes:
        snap = strike_dfs[s][strike_dfs[s]["_mins"] <= SCAN_MINS]
        if snap.empty: continue
        r = snap.iloc[-1]
        baseline_920[int(s)] = {
            "spot": float(r["synthetic_spot"]),  "straddle": float(r["straddle_price"]),
            "ce_oi": float(r["ce_oi"]),           "pe_oi":    float(r["pe_oi"]),
        }
    if not baseline_920:
        return {"date": date_str, "day_pnl": 0.0, "trades": [], "strike_selected": None,
                "days_to_expiry": (expiry - trade_date).days}

    # ── Session state (local, resets each function call) ─────
    session_pnl          = 0.0
    trades_taken         = 0
    strike_last_pnl      = {s: 0.0 for s in strikes}
    strike_consec_losses = {s: 0   for s in strikes}
    blacklist            = {}   # {strike: blacklist_expiry_mins}
    current_mins         = ENTRY_MINS
    day_trades           = []
    first_strike         = None

    while current_mins < HARD_EXIT_M and trades_taken < MAX_TRADES:
        if session_pnl <= DAILY_SL: break

        # Snapshot at current_mins (past data only)
        snap_at_t = {}
        for s in strikes:
            sd = strike_dfs[s]
            snap = sd[sd["_mins"] <= current_mins]
            if snap.empty: continue
            snap_at_t[s] = snap.iloc[-1]
        if len(snap_at_t) < 3:
            current_mins += 1; continue

        vwap_map  = {s: float(snap_at_t[s]["vwap"]) if float(snap_at_t[s]["vwap"]) > 0
                     else float(snap_at_t[s]["straddle_price"]) for s in snap_at_t}
        available = [s for s in snap_at_t if blacklist.get(s, 0) <= current_mins]
        if not available: available = list(snap_at_t.keys())

        # Krishna V3 strike selection
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
                snapshot_df=snap_df,
                vwap_map={s: vwap_map[s] for s in available},
                baseline_920={int(s): baseline_920[int(s)] for s in available if int(s) in baseline_920},
                current_time=dt.time(current_mins // 60, current_mins % 60),
                vix=vix, nifty_prev_change=n_prev, nifty_open_gap=n_gap,
                trade_date=trade_date, expiry_date=expiry, model=m_v3,
                session_pnl=session_pnl, trades_taken=trades_taken,
                strike_last_pnl={s: strike_last_pnl.get(s, 0.0) for s in available},
                strike_consec_losses={s: strike_consec_losses.get(s, 0) for s in available},
            )
            chosen = int(pred["best_strike"])
            v3_conf = pred["confidence"]
        except Exception:
            chosen  = available[len(available) // 2]
            v3_conf = 0.0

        if first_strike is None:
            first_strike = chosen

        # Check VWAP entry condition  
        chosen_row  = snap_at_t.get(chosen)
        if chosen_row is None:
            current_mins += 5; continue

        entry_px    = float(chosen_row["straddle_price"])
        entry_vwap  = vwap_map.get(chosen, entry_px)
        if entry_px > entry_vwap * (1.0 - threshold / 100.0):
            current_mins += 1; continue

        # Trade simulation (minute-by-minute, no future data)
        arr      = strike_arrs.get(chosen)
        if arr is None:
            current_mins += 5; continue
        idx      = int(np.searchsorted(arr[:, 2], current_mins))
        if idx >= len(arr) - 3:
            current_mins += 5; continue

        sl_px        = entry_px * SL_MULT
        max_pnl_pts  = 0.0
        is_safe_mode = False
        entry_iv     = float(chosen_row.get("ce_iv") or 0) + float(chosen_row.get("pe_iv") or 0)
        vol_window   = []
        pnl_pts      = 0.0
        exit_reason  = "EOD"
        exit_mins_v  = HARD_EXIT_M
        sd_chosen    = strike_dfs[chosen]

        for j in range(idx, len(arr)):
            px     = arr[j, 0]
            vwap_j = arr[j, 1] if arr[j, 1] > 0 else px
            mins_j = int(arr[j, 2])
            pnl_pts = entry_px - px

            if mins_j >= HARD_EXIT_M:
                exit_reason = "HARD_EXIT"; exit_mins_v = mins_j; break
            if px >= sl_px:
                exit_reason = "SL"; exit_mins_v = mins_j; break

            # Break-Even SL
            if not is_safe_mode and pnl_pts >= BE_SL_ON:
                is_safe_mode = True
            if is_safe_mode and pnl_pts <= BE_SL_OFF:
                exit_reason = "BE_SL"; exit_mins_v = mins_j; break

            max_pnl_pts = max(max_pnl_pts, pnl_pts)
            drawdown    = max_pnl_pts - pnl_pts

            row_j = sd_chosen[sd_chosen["_mins"] == mins_j]
            if not row_j.empty:
                r = row_j.iloc[-1]
                vol = float(r.get("ce_volume") or 0) + float(r.get("pe_volume") or 0)
                vol_window.append(vol)
                if len(vol_window) > 15: vol_window.pop(0)
                avg_vol  = max(sum(vol_window) / len(vol_window), 1.0)
                rel_vol  = min(vol / avg_vol, 100.0)
                curr_iv  = float(r.get("ce_iv") or 0) + float(r.get("pe_iv") or 0)
                iv_d     = curr_iv - entry_iv
                delta_d  = abs(float(r.get("ce_delta") or 0) + float(r.get("pe_delta") or 0))
                theta_v  = float(r.get("ce_theta") or 0) + float(r.get("pe_theta") or 0)
                vwap_gap = (vwap_j - px) / vwap_j * 100 if vwap_j else 0
                try:
                    res = v4.predict_exit_v4(
                        pnl_pts=pnl_pts, max_pnl_so_far=max_pnl_pts, drawdown=drawdown,
                        vwap_gap_pct=vwap_gap, delta_drift=delta_d,
                        theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                        session_pnl=session_pnl, trades_taken=trades_taken,
                        is_safe_mode=is_safe_mode, model=m_v4, threshold=EXIT_THRESH
                    )
                    if res["should_exit"]:
                        exit_reason = "ARJUN_V4"; exit_mins_v = mins_j; break
                except Exception:
                    pass

            if px > vwap_j:
                exit_reason = "VWAP_CROSS"; exit_mins_v = mins_j; break

        # Update session state
        session_pnl += pnl_pts
        trades_taken += 1
        strike_last_pnl[chosen] = pnl_pts
        if pnl_pts < 0:
            strike_consec_losses[chosen] = strike_consec_losses.get(chosen, 0) + 1
            if pnl_pts <= -7.0:
                blacklist[chosen] = current_mins + 60
        else:
            strike_consec_losses[chosen] = 0

        entry_t = dt.time(current_mins // 60, current_mins % 60)
        exit_t  = dt.time(min(exit_mins_v, HARD_EXIT_M) // 60, min(exit_mins_v, HARD_EXIT_M) % 60)
        day_trades.append({
            "strike":       chosen,
            "entry_time":   str(entry_t)[:5],
            "exit_time":    str(exit_t)[:5],
            "entry_price":  round(entry_px, 2),
            "exit_price":   round(entry_px - pnl_pts, 2),
            "pnl_pts":      round(pnl_pts, 2),
            "exit_reason":  exit_reason,
            "v3_conf":      round(v3_conf, 3),
            "safe_mode_hit":is_safe_mode,
            "trade_num":    trades_taken,
        })

        cooldown     = COOLDOWN_M if exit_reason == "SL" else 5
        current_mins = exit_mins_v + cooldown

    day_pnl = sum(t["pnl_pts"] for t in day_trades)
    return {
        "date":         date_str,
        "day_pnl":      round(day_pnl, 2),
        "trades":       day_trades,
        "trade_count":  len(day_trades),
        "first_strike": first_strike,
        "days_to_exp":  (expiry - trade_date).days,
        "vix":          vix,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V4 Realistic Backtest")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="VWAP entry threshold %% (default: 0.2)")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    args = parser.parse_args()

    t0 = timer.time()
    print(f"\n🔱 V4 Realistic Backtest — Krishna V3 + Arjun V4")
    print(f"   Threshold : {args.threshold}%")
    print(f"   Period    : {args.start} to {args.end}")
    print(f"   SL Mult   : {SL_MULT}x | Max Trades/Day: {MAX_TRADES}\n")

    print("   📥 Loading candles from DB...")
    t_db = timer.time()
    with db.get_conn() as conn:
        candles = pd.read_sql(
            f"SELECT trade_date, expiry, strike, atm, ts, straddle_price, vwap, "
            f"ce_close, pe_close, ce_oi, pe_oi, synthetic_spot, ce_volume, pe_volume, "
            f"ce_iv, pe_iv, ce_delta, pe_delta, ce_theta, pe_theta, "
            f"ce_gamma, pe_gamma, ce_vega, pe_vega "
            f"FROM straddle_candles "
            f"WHERE trade_date >= '{args.start}' AND trade_date <= '{args.end}' "
            f"ORDER BY trade_date, strike, ts",
            conn
        )
        mc_df = pd.read_sql(
            f"SELECT trade_date, vix, nifty_prev_day_change, nifty_open_gap "
            f"FROM market_context "
            f"WHERE trade_date >= '{args.start}' AND trade_date <= '{args.end}'",
            conn
        )
    print(f"   ✅ {len(candles):,} candles, {mc_df['trade_date'].nunique()} days in {timer.time()-t_db:.0f}s")

    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"]    = candles["ts"].dt.tz_convert(IST)
    candles["_mins"] = (candles["ts"].dt.hour * 60 + candles["ts"].dt.minute).astype(np.int16)

    for col in ["straddle_price","vwap","ce_close","pe_close","ce_oi","pe_oi",
                "synthetic_spot","ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta",
                "ce_gamma","pe_gamma","ce_vega","pe_vega","ce_volume","pe_volume","atm"]:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0.0)

    mc_map = {str(r["trade_date"]): {
        "vix": float(r.get("vix") or 0),
        "nifty_prev_day_change": float(r.get("nifty_prev_day_change") or 0),
        "nifty_open_gap": float(r.get("nifty_open_gap") or 0),
    } for _, r in mc_df.iterrows()}

    for f in [MODEL_V3, MODEL_V4]:
        if not os.path.exists(f):
            print(f"❌ Model not found: {f}"); return

    keep = ["trade_date","expiry","strike","atm","straddle_price","vwap","_mins",
            "ce_close","pe_close","ce_oi","pe_oi","synthetic_spot",
            "ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta",
            "ce_gamma","pe_gamma","ce_vega","pe_vega","ce_volume","pe_volume"]

    days = sorted(candles["trade_date"].unique())
    pool_args = []
    for d in days:
        date_str = str(d)
        mc       = mc_map.get(date_str, {"vix": 0, "nifty_prev_day_change": 0, "nifty_open_gap": 0})
        pool_args.append((
            date_str,
            candles[candles["trade_date"] == d][keep].to_dict("list"),
            mc["vix"], mc["nifty_prev_day_change"], mc["nifty_open_gap"],
            args.threshold, MODEL_V3, MODEL_V4
        ))

    cores = max(1, os.cpu_count() - 2)
    print(f"\n   🔥 Running {len(days)} days on {cores} cores...")
    t1 = timer.time()

    day_results = []
    with mp.Pool(processes=cores) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_worker_day, pool_args, chunksize=max(1, len(pool_args) // (cores * 4))), 1
        ):
            day_results.append(res)
            if i % 30 == 0 or i == len(days):
                elapsed = timer.time() - t1
                print(f"   [{i:3d}/{len(days)}] {i/len(days):.0%} | Elapsed: {elapsed:.0f}s")

    print(f"\n   ✅ Backtest complete in {timer.time()-t1:.0f}s")

    # Sort by date
    day_results = sorted(day_results, key=lambda x: x["date"])

    # Build per-trade flat table
    trade_rows = []
    for day in day_results:
        for t in day["trades"]:
            trade_rows.append({
                "date":        day["date"],
                "vix":         day["vix"],
                **t,
                "session_pnl_after": round(
                    sum(x["pnl_pts"] for x in day["trades"][:day["trades"].index(t)+1]), 2
                ) if t in day["trades"] else 0,
            })

    trade_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame([{
        "date":          d["date"],
        "day_pnl":       d["day_pnl"],
        "trade_count":   d["trade_count"],
        "first_strike":  d["first_strike"],
        "vix":           d["vix"],
        "days_to_exp":   d["days_to_exp"],
    } for d in day_results])

    # Summary statistics
    pnl_arr = daily_df["day_pnl"].tolist()
    total   = sum(pnl_arr)
    wins    = len([p for p in pnl_arr if p > 0])
    losses  = len([p for p in pnl_arr if p < 0])
    flat    = len([p for p in pnl_arr if p == 0])
    wr      = wins / len(pnl_arr) * 100 if pnl_arr else 0
    avg_win = np.mean([p for p in pnl_arr if p > 0]) if wins else 0
    avg_loss= np.mean([p for p in pnl_arr if p < 0]) if losses else 0
    cum     = np.cumsum(pnl_arr)
    max_dd  = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0

    exit_counts = trade_df["exit_reason"].value_counts().to_dict() if not trade_df.empty else {}

    print(f"\n{'='*65}")
    print(f"  🔱 V4 BACKTEST RESULTS (Threshold: {args.threshold}%)")
    print(f"{'='*65}")
    print(f"  Period       : {args.start} to {args.end}")
    print(f"  Trading Days : {len(pnl_arr)}")
    print(f"  Total P&L    : {total:+.1f} pts")
    print(f"  Win Rate     : {wr:.1f}% ({wins}W / {losses}L / {flat}F)")
    print(f"  Avg Win Day  : {avg_win:+.1f} pts")
    print(f"  Avg Loss Day : {avg_loss:+.1f} pts")
    print(f"  Max Drawdown : {max_dd:.1f} pts")
    print(f"  Avg/Day      : {total/len(pnl_arr):+.1f} pts")
    print(f"  Total Trades : {len(trade_df)}")
    print(f"  Exit Reasons : {exit_counts}")
    print(f"{'='*65}\n")

    # Save reports
    tag = f"{args.start}_to_{args.end}_{args.threshold}"
    daily_out = os.path.join(config.LOGS_PATH, f"v4_backtest_daily_{tag}.csv")
    trade_out = os.path.join(config.LOGS_PATH, f"v4_backtest_trades_{tag}.csv")

    daily_df.to_csv(daily_out, index=False)
    if not trade_df.empty:
        trade_df.to_csv(trade_out, index=False)

    print(f"📊 Daily report  → {daily_out}")
    print(f"📊 Trades report → {trade_out}")
    print(f"⏱️  Total time    : {timer.time()-t0:.0f}s\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
