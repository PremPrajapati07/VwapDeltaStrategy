"""
Backtest: Krishna V1 + Arjun (Multiprocessing Optimized)

Loads data once, distributes days across CPU cores for parallel simulation.
Saves both daily summary and detailed trade-by-trade logs.
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
import krishna_model as v1
import arjun_model as wa

IST = ZoneInfo("Asia/Kolkata")


def _process_day(args):
    """Worker: process one day using V1 strike selection + Arjun exits."""
    day_dict, mc, trade_dt_str, expiry_str = args
    import config as cfg
    import arjun_model as wa
    import krishna_model as wv1

    day_candles = pd.DataFrame(day_dict)
    if len(day_candles) < 200:
        return None

    arjun = wa.load_arjun_model()
    # V1 model is loaded in the worker
    try:
        krishna = wv1.load_model()
    except:
        return None

    trade_dt = dt.date.fromisoformat(trade_dt_str)
    expiry_date = dt.date.fromisoformat(expiry_str)

    # Build snapshot at 9:20 for V1 prediction
    snap_920 = day_candles[day_candles["_mins"] <= 9 * 60 + 20]
    if snap_920.empty:
        return None

    latest_920 = snap_920.groupby("strike").last().reset_index()
    latest_920 = latest_920.rename(columns={
        "ce_close": "ce_ltp", 
        "pe_close": "pe_ltp",
        "ce_oi": "oi_ce",
        "pe_oi": "oi_pe"
    })

    vwap_map = {}
    for _, r in latest_920.iterrows():
        vwap_map[int(r["strike"])] = float(r["vwap"]) if float(r["vwap"]) > 0 else float(r["straddle_price"])

    v1_result = wv1.predict_best_strike(
        snapshot_df=latest_920,
        vix=mc["vix"],
        nifty_prev_change=mc["nifty_prev_day_change"],
        nifty_open_gap=mc["nifty_open_gap"],
        vwap_map=vwap_map,
        trade_date=trade_dt,
        expiry_date=expiry_date
    )
    selected_strike = v1_result["best_strike"]

    strike_candles = day_candles[day_candles["strike"] == selected_strike].copy()
    if strike_candles.empty:
        return None

    # Rename for Arjun compatibility
    strike_candles = strike_candles.rename(columns={"_ts_ist": "datetime"})

    day_trades = wa.simulate_pnl_with_arjun_v2(
        strike_candles, arjun_model=arjun,
        threshold=cfg.ARJUN_EXIT_THRESHOLD,
        p_target=cfg.DAILY_PROFIT_TARGET,
        s_loss=cfg.DAILY_STOP_LOSS,
    )

    # Add metadata to each trade for flattening
    for t in day_trades:
        t["trade_date"] = trade_dt_str
        t["v1_strike"] = selected_strike
        t["v1_confidence"] = round(v1_result["confidence"], 4)
        if "entry_time" in t:
            t["entry_time"] = t["entry_time"].strftime('%H:%M:%S') if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        if "exit_time" in t:
            t["exit_time"] = t["exit_time"].strftime('%H:%M:%S') if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])

    day_pnl = sum(t["pnl_pts"] for t in day_trades)

    return {
        "summary": {
            "trade_date": trade_dt_str,
            "v1_strike": selected_strike,
            "v1_confidence": round(v1_result["confidence"], 4),
            "pnl": round(day_pnl, 1),
            "trades": len(day_trades),
        },
        "trades": day_trades
    }


def run_v1_backtest(start_date="2025-01-01", end_date="2025-12-31"):
    print(f"\n🔱 Krishna V1 + Arjun Backtest ({start_date} → {end_date})")
    print(f"   VWAP Threshold: {config.VWAP_ENTRY_THRESHOLD_PCT}%")
    print(f"   Daily SL: {config.DAILY_STOP_LOSS}\n")

    t0 = timer.time()

    cols_to_fetch = [
        "trade_date", "strike", "atm", "straddle_price", "vwap", 
        "ce_close", "pe_close", "ce_oi", "pe_oi", "ce_volume", "pe_volume", 
        "synthetic_spot", "ce_iv", "pe_iv", "ce_delta", "pe_delta", 
        "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", "ce_vega", "pe_vega", 
        "vwap_gap_pct", "ts", "expiry"
    ]
    cols_str = ", ".join(cols_to_fetch)

    with db.get_conn() as conn:
        q_feat = f"SELECT * FROM daily_features WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
        q_cand = f"SELECT {cols_str} FROM straddle_candles WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
        feat_df = pd.read_sql(q_feat, conn)
        print(f"   Fetching ~2.7M rows from DB...")
        candles = pd.read_sql(q_cand + " ORDER BY trade_date, strike, ts", conn)

    print(f"   Data loaded in {timer.time() - t0:.1f}s. Candles: {len(candles):,}")

    if candles.empty:
        print("❌ No data.")
        return

    # Timezone
    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"] = candles["ts"].dt.tz_convert(IST)
    candles["_mins"] = candles["ts"].dt.hour * 60 + candles["ts"].dt.minute
    candles["_ts_ist"] = candles["ts"]

    num_cols = [
        "ce_iv", "pe_iv", "ce_delta", "pe_delta", "ce_theta", "pe_theta",
        "ce_gamma", "pe_gamma", "ce_vega", "pe_vega", "synthetic_spot",
        "vwap", "vwap_gap_pct", "straddle_price", "ce_close", "pe_close",
        "ce_oi", "pe_oi", "atm", "ce_volume", "pe_volume"
    ]
    for col in num_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    # Build per-day args
    days = sorted(candles["trade_date"].unique())
    pool_args = []

    keep_cols = ["trade_date", "strike", "atm", "straddle_price", "vwap",
                 "ce_close", "pe_close", "ce_oi", "pe_oi", "ce_volume", "pe_volume",
                 "synthetic_spot", "ce_iv", "pe_iv", "ce_delta", "pe_delta",
                 "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", "ce_vega", "pe_vega",
                 "vwap_gap_pct", "_mins", "_ts_ist", "expiry"]

    for trade_date in days:
        date_str = str(trade_date)
        day_data = candles[candles["trade_date"] == trade_date][keep_cols].copy()

        if len(day_data) < 200:
            continue

        day_feat = feat_df[pd.to_datetime(feat_df["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        mc = {
            "vix": float(day_feat.iloc[0].get("vix") or 0) if not day_feat.empty else 0,
            "nifty_prev_day_change": float(day_feat.iloc[0].get("nifty_prev_day_change") or 0) if not day_feat.empty else 0,
            "nifty_open_gap": float(day_feat.iloc[0].get("nifty_open_gap") or 0) if not day_feat.empty else 0,
        }

        expiry = day_data.iloc[0]["expiry"]
        expiry_str = str(pd.to_datetime(expiry).date() if isinstance(expiry, str) else expiry)
        trade_dt_str = str(pd.to_datetime(trade_date).date() if isinstance(trade_date, str) else trade_date)

        pool_args.append((day_data.to_dict("list"), mc, trade_dt_str, expiry_str))

    cores = max(1, os.cpu_count() - 2)
    print(f"   🚀 Parallel V1 backtest starting ({len(pool_args)} days)...")

    t1 = timer.time()
    with mp.Pool(processes=cores) as pool:
        raw_results = pool.map(_process_day, pool_args)

    daily_summary = []
    detailed_trades = []
    for r in raw_results:
        if r is not None:
            daily_summary.append(r["summary"])
            detailed_trades.extend(r["trades"])

    print(f"   ✅ Simulation complete in {timer.time() - t1:.1f}s")

    if not daily_summary:
        print("❌ No results.")
        return

    # 1. Save Detailed Trades
    detailed_df = pd.DataFrame(detailed_trades)
    detailed_path = os.path.join(config.LOGS_PATH, f"v1_backtest_detailed_trades_{start_date}_to_{end_date}_{config.VWAP_ENTRY_THRESHOLD_PCT}.csv")
    if not detailed_df.empty:
        cols = ["trade_date", "v1_strike", "v1_confidence", "entry_time", "entry_price", "exit_time", "exit_price", "pnl_pts", "exit_reason"]
        detailed_df = detailed_df[[c for c in cols if c in detailed_df.columns]]
        detailed_df.to_csv(detailed_path, index=False)
        print(f"   📄 Detailed Record → {detailed_path}")

    # 2. Save Daily Summary
    summary_df = pd.DataFrame(daily_summary)
    summary_path = os.path.join(config.LOGS_PATH, f"v1_backtest_daily_report_{start_date}_to_{end_date}_{config.VWAP_ENTRY_THRESHOLD_PCT}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"   📄 Daily Summary   → {summary_path}")

    total_pnl = summary_df["pnl"].sum()
    print("\n" + "=" * 65)
    print(f"   🔱 V1 TOTAL P&L (2025): {total_pnl:,.1f} pts")
    print(f"   🔱 V1 Win Rate: {len(summary_df[summary_df['pnl']>0])/len(summary_df):.0%}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    run_v1_backtest(args.start, args.end)
