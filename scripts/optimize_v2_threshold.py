"""
Optimization Sweep: VWAP_ENTRY_THRESHOLD_PCT (0.0% to 2.0%)
Model: Krishna V2 + Arjun V3 (Break-Even SL)
Year: 2025
"""

import os
import sys
import datetime as dt
import time as timer
import pandas as pd
import numpy as np
import multiprocessing as mp
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db
import krishna_v2_model as v2
import arjun_model_v3 as wa

IST = ZoneInfo("Asia/Kolkata")
THRESHOLDS = [round(x * 0.1, 1) for x in range(0, 21)]
START_DATE = "2025-01-01"
END_DATE   = "2025-12-31"


def simulate_with_threshold(grp, arjun_model, entry_threshold, exit_threshold=0.65, p_target=100.0, s_loss=-50.0):
    import config as cfg
    import arjun_model_v3 as wa_local
    hard_exit      = dt.time(15, 0)
    trades         = []
    position       = False
    entry_price    = 0.0
    entry_iv       = 0.0
    max_pnl        = 0.0
    vol_window     = []
    cooldown_until = None
    cumulative_pnl = 0.0
    trade_count    = 0
    is_safe_mode   = False

    for _, row in grp.iterrows():
        ts_full = row["_ts_ist"] if "_ts_ist" in row.index else row["datetime"]
        t       = ts_full.time() if hasattr(ts_full, "time") else ts_full
        price   = float(row["straddle_price"])
        vwap    = float(row["vwap"]) if float(row["vwap"]) > 0 else price

        if not position:
            if trade_count >= cfg.MAX_TRADES_PER_DAY: break
            if cumulative_pnl <= s_loss: break
            if trade_count >= cfg.MAX_TRADES_FOR_PROFIT_TARGET and cumulative_pnl >= p_target: break

        if cooldown_until and ts_full < cooldown_until:
            continue

        if t >= hard_exit:
            if position:
                pnl_pts = entry_price - price
                trades.append({"pnl_pts": pnl_pts, "exit_reason": "HARD_EXIT"})
                cumulative_pnl += pnl_pts
                trade_count += 1
                position = False
            break

        if not position:
            if price <= vwap * (1.0 - entry_threshold / 100.0):
                entry_price  = price
                entry_iv     = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
                max_pnl      = 0.0
                vol_window   = []
                position     = True
                is_safe_mode = False
        else:
            pnl_pts  = entry_price - price
            max_pnl  = max(max_pnl, pnl_pts)
            drawdown = max_pnl - pnl_pts
            vol      = float(row.get("ce_volume", 0)) + float(row.get("pe_volume", 0))
            vol_window.append(vol)
            if len(vol_window) > 15: vol_window.pop(0)
            avg_vol  = max(sum(vol_window) / len(vol_window), 1.0)
            rel_vol  = min(vol / avg_vol, 100.0)

            curr_iv  = float(row.get("ce_iv", 0)) + float(row.get("pe_iv", 0))
            delta_d  = abs(float(row.get("ce_delta", 0)) + float(row.get("pe_delta", 0)))
            theta_v  = float(row.get("ce_theta", 0)) + float(row.get("pe_theta", 0))
            vwap_gap = (vwap - price) / vwap * 100 if vwap else 0
            iv_d     = curr_iv - entry_iv

            # Break-Even Trailing SL (+30 pts -> +2 pts)
            if not is_safe_mode and pnl_pts >= 30.0:
                is_safe_mode = True

            decision = wa_local.predict_exit(
                pnl_pts=pnl_pts, max_pnl_so_far=max_pnl, drawdown=drawdown,
                vwap_gap_pct=vwap_gap, delta_drift=delta_d,
                theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                model=arjun_model, threshold=exit_threshold,
            )

            exit_reason = None
            if is_safe_mode and pnl_pts <= 2.0:
                exit_reason = "BE_SL_TRIGGERED"
            elif decision["should_exit"]:
                exit_reason = "ARJUN_EXIT"
            elif price >= vwap:
                exit_reason = "VWAP_CROSS"

            if exit_reason:
                trades.append({"pnl_pts": pnl_pts, "exit_reason": exit_reason})
                cumulative_pnl += pnl_pts
                trade_count    += 1
                position        = False
                is_safe_mode    = False
                cooldown_until  = ts_full + pd.Timedelta(minutes=cfg.SL_COOLDOWN_MINUTES)
    return trades


def _worker_threshold_sweep(args):
    threshold, day_to_strike_map, day_to_candles_map = args
    import arjun_model_v3 as wa_w
    import config as cfg

    arjun       = wa_w.load_arjun_model()
    all_results = []

    for date_str, candles_dict in day_to_candles_map.items():
        selected_strike = day_to_strike_map.get(date_str)
        if selected_strike is None:
            continue
        strike_candles = pd.DataFrame(candles_dict)
        res     = simulate_with_threshold(
            strike_candles, arjun,
            entry_threshold=threshold,
            exit_threshold=cfg.ARJUN_EXIT_THRESHOLD,
            p_target=cfg.DAILY_PROFIT_TARGET,
            s_loss=cfg.DAILY_STOP_LOSS
        )
        day_pnl = sum(t["pnl_pts"] for t in res)
        all_results.append({"pnl": day_pnl, "trades": len(res)})

    pnl_arr     = [r["pnl"] for r in all_results]
    trade_count = sum(r["trades"] for r in all_results)
    total_pnl   = sum(pnl_arr)
    win_days    = len([p for p in pnl_arr if p > 0])
    win_rate    = (win_days / len(pnl_arr)) * 100 if pnl_arr else 0
    cum_pnl     = np.cumsum(pnl_arr)
    max_dd      = (cum_pnl - np.maximum.accumulate(cum_pnl)).min() if len(cum_pnl) > 0 else 0

    return {
        "Threshold (%)": f"{threshold:.1f}%",
        "P&L (pts)":     round(total_pnl, 1),
        "Win Rate":      f"{win_rate:.1f}%",
        "Trades":        trade_count,
        "Max DD":        round(max_dd, 1),
        "Avg/Day":       round(total_pnl / len(pnl_arr), 1) if pnl_arr else 0
    }


def run_optimization():
    print(f"\n🚀 Krishna V2 + Arjun V3 Optimization Scan [0.0% → 2.0%]")
    print(f"   Break-Even SL: ON (+30 pts → +2 pts trailing)")
    print(f"   Period: {START_DATE} to {END_DATE}\n")

    t0 = timer.time()
    with db.get_conn() as conn:
        feat_df = pd.read_sql(
            f"SELECT * FROM daily_features WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}'", conn)
        print("   Loading candles (~2.7M rows)...")
        candles = pd.read_sql(
            f"SELECT * FROM straddle_candles WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}' ORDER BY trade_date, strike, ts", conn)

    print(f"   ✅ Data loaded in {timer.time() - t0:.1f}s")

    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"]    = candles["ts"].dt.tz_convert(IST)
    candles["_mins"] = candles["ts"].dt.hour * 60 + candles["ts"].dt.minute
    candles["_ts_ist"] = candles["ts"]

    for col in ["ce_iv","pe_iv","ce_delta","pe_delta","ce_theta","pe_theta","ce_gamma","pe_gamma",
                "ce_vega","pe_vega","synthetic_spot","vwap","straddle_price","ce_close","pe_close",
                "ce_oi","pe_oi","atm","ce_volume","pe_volume"]:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    print("   🎯 Pre-calculating Krishna V2 strike for each day...")
    day_to_strike       = {}
    day_to_candles_full = {}
    krishna = v2.load_v2_model()

    for trade_date in sorted(candles["trade_date"].unique()):
        date_str = str(trade_date)
        day_data = candles[candles["trade_date"] == trade_date].copy()
        snap_925 = day_data[day_data["_mins"] <= 9 * 60 + 25].copy()
        if snap_925.empty: continue

        latest = snap_925.groupby("strike").last().reset_index()
        latest = latest.rename(columns={"ce_close": "ce_ltp", "pe_close": "pe_ltp"})
        vwap_map = {int(r["strike"]): float(r["vwap"]) if float(r["vwap"]) > 0 else float(r["straddle_price"])
                    for _, r in latest.iterrows()}

        baseline_920 = {}
        for strike in latest["strike"].unique():
            sd = day_data[day_data["strike"] == strike]
            snap_920 = sd[sd["_mins"] <= 9 * 60 + 20]
            if snap_920.empty: continue
            r = snap_920.iloc[-1]
            baseline_920[int(strike)] = {
                "spot": float(r["synthetic_spot"]), "straddle": float(r["straddle_price"]),
                "ce_oi": float(r["ce_oi"]),         "pe_oi":    float(r["pe_oi"]),
            }

        day_feat   = feat_df[pd.to_datetime(feat_df["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        vix        = float(day_feat.iloc[0].get("vix") or 0)        if not day_feat.empty else 0
        n_prev_chg = float(day_feat.iloc[0].get("nifty_prev_day_change") or 0) if not day_feat.empty else 0
        n_open_gap = float(day_feat.iloc[0].get("nifty_open_gap") or 0)        if not day_feat.empty else 0

        expiry    = day_data.iloc[0]["expiry"]
        expiry_dt = pd.to_datetime(expiry).date()

        v2_res = v2.predict_best_strike_v2(
            latest, vwap_map, baseline_920, dt.time(9, 25),
            vix, n_prev_chg, n_open_gap, trade_date, expiry_dt, model=krishna
        )
        best_strike             = v2_res["best_strike"]
        day_to_strike[date_str] = best_strike
        day_to_candles_full[date_str] = day_data[day_data["strike"] == best_strike].to_dict("list")

    print(f"   🔥 Pre-calc done. Launching parallel sweep on {len(THRESHOLDS)} thresholds...")
    pool_args = [(th, day_to_strike, day_to_candles_full) for th in THRESHOLDS]
    cores = max(1, os.cpu_count() - 2)

    t1 = timer.time()
    with mp.Pool(processes=cores) as pool:
        final_results = pool.map(_worker_threshold_sweep, pool_args)
    print(f"   ✅ Sweep complete in {timer.time() - t1:.1f}s")

    results_df = pd.DataFrame(final_results)
    print("\n" + "=" * 80)
    print("   🔱 KRISHNA V2 + ARJUN V3 THRESHOLD OPTIMIZATION (2025)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    best_row = results_df.loc[pd.to_numeric(results_df["P&L (pts)"]).idxmax()]
    print(f"\n🏆 WINNER: {best_row['Threshold (%)']} → {best_row['P&L (pts)']} pts | {best_row['Win Rate']} Win Rate")

    out_path = os.path.join(config.LOGS_PATH, "optimization_v3_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"📊 Results saved → {out_path}\n")


if __name__ == "__main__":
    run_optimization()
