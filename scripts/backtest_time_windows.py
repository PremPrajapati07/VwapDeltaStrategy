"""
Time-Window Sweep Backtest — Krishna V2

Tests the same V2 model across multiple start-time windows to find
the optimal entry window. For each window:
  - Strike is selected using V2 at (window_start + 5 min)
  - Only candles >= window_start_mins are considered for entry
  - Hard exit at 15:00 IST always

Windows tested (configurable):
  9:15, 9:20, 10:00, 11:00, 11:30, 12:00

Output: comparison CSV + Excel + printed table
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

IST = ZoneInfo("Asia/Kolkata")

# ── Windows to test ─────────────────────────────────────────
# Each entry: (label, start_mins, selection_mins, max_trades)
# start_mins = when entries are allowed from
# selection_mins = when V2 picks the strike (start + 5)
WINDOWS = [
    ("09:15 (baseline)", 9 * 60 + 15, 9 * 60 + 20, 5),
    ("09:20",            9 * 60 + 20, 9 * 60 + 25, 5),
    ("10:00",           10 * 60 + 0, 10 * 60 + 5,  4),
    ("11:00",           11 * 60 + 0, 11 * 60 + 5,  3),
    ("11:30",           11 * 60 + 30, 11 * 60 + 35, 3),
    ("12:00",           12 * 60 + 0, 12 * 60 + 5,  2),
]

HARD_EXIT_MINS = 15 * 60   # 15:00 IST
VWAP_THRESHOLD_PCT = config.VWAP_ENTRY_THRESHOLD_PCT
DAILY_SL = config.DAILY_STOP_LOSS


# ── Fast inline simulator (no Arjun model, pure VWAP strategy) ──
def _simulate_window(candles_arr, start_mins, max_trades):
    """
    Vectorized VWAP-based simulation for a single strike's candle data.
    candles_arr: list of (mins, straddle_price, vwap, ts_str)
    Returns list of trade dicts.
    """
    threshold_factor = 1.0 - VWAP_THRESHOLD_PCT / 100.0
    trades = []
    position = False
    entry_price = 0.0
    entry_time = ""
    cumulative_pnl = 0.0
    trade_count = 0

    for (mins, price, vwap, ts_str) in candles_arr:
        if mins < start_mins:
            continue
        if mins >= HARD_EXIT_MINS:
            if position:
                pnl = entry_price - price
                trades.append({"entry_time": entry_time, "entry_price": entry_price,
                                "exit_time": ts_str, "exit_price": price,
                                "pnl_pts": round(pnl, 2), "exit_reason": "HARD_EXIT"})
                cumulative_pnl += pnl
                trade_count += 1
                position = False
            break

        if not position:
            if trade_count >= max_trades:
                break
            if cumulative_pnl <= DAILY_SL:
                break
            eff_vwap = vwap if vwap > 0 else price
            if price <= eff_vwap * threshold_factor:
                entry_price = price
                entry_time = ts_str
                position = True
        else:
            eff_vwap = vwap if vwap > 0 else price
            if price > eff_vwap:
                pnl = entry_price - price
                trades.append({"entry_time": entry_time, "entry_price": entry_price,
                                "exit_time": ts_str, "exit_price": price,
                                "pnl_pts": round(pnl, 2), "exit_reason": "VWAP_CROSS"})
                cumulative_pnl += pnl
                trade_count += 1
                position = False

    # Still open at end of loop
    if position and candles_arr:
        last = candles_arr[-1]
        pnl = entry_price - last[1]
        trades.append({"entry_time": entry_time, "entry_price": entry_price,
                        "exit_time": last[3], "exit_price": last[1],
                        "pnl_pts": round(pnl, 2), "exit_reason": "EOD"})

    return trades


# ── Per-day Worker ─────────────────────────────────────────
def _process_day_windows(args):
    """Process one day across ALL windows. Returns list of {window, ...} dicts."""
    day_dict, baseline_dict, mc, trade_dt_str, expiry_str = args

    import krishna_v2_model as wv2

    day_candles = pd.DataFrame(day_dict)
    if len(day_candles) < 100:
        return []

    krishna = wv2.load_v2_model()
    trade_dt = dt.date.fromisoformat(trade_dt_str)
    expiry_date = dt.date.fromisoformat(expiry_str)

    # Pre-build a tuple list per strike for fast simulation: (mins, price, vwap, ts_str)
    strike_tuples = {}
    for strike in day_candles["strike"].unique():
        sd = day_candles[day_candles["strike"] == strike].sort_values("_mins")
        arr = list(zip(
            sd["_mins"].tolist(),
            sd["straddle_price"].astype(float).tolist(),
            sd["vwap"].astype(float).tolist(),
            sd["_ts_str"].tolist(),
        ))
        strike_tuples[int(strike)] = arr

    day_results = []

    for (label, start_mins, sel_mins, max_trades) in WINDOWS:
        # V2 strike selection at sel_mins
        snap = day_candles[day_candles["_mins"] <= sel_mins]
        if snap.empty:
            continue

        latest = snap.groupby("strike").last().reset_index()
        latest = latest.rename(columns={"ce_close": "ce_ltp", "pe_close": "pe_ltp"})

        vwap_map = {}
        for _, r in latest.iterrows():
            vwap_map[int(r["strike"])] = float(r["vwap"]) if float(r["vwap"]) > 0 else float(r["straddle_price"])

        try:
            v2_result = wv2.predict_best_strike_v2(
                snapshot_df=latest, vwap_map=vwap_map, baseline_920=baseline_dict,
                current_time=dt.time(sel_mins // 60, sel_mins % 60),
                vix=mc["vix"], nifty_prev_change=mc["nifty_prev_day_change"],
                nifty_open_gap=mc["nifty_open_gap"],
                trade_date=trade_dt, expiry_date=expiry_date, model=krishna,
            )
        except Exception:
            continue

        selected_strike = v2_result["best_strike"]
        candle_arr = strike_tuples.get(selected_strike, [])
        if not candle_arr:
            continue

        trades = _simulate_window(candle_arr, start_mins, max_trades)
        day_pnl = sum(t["pnl_pts"] for t in trades)

        # Tag each trade with metadata
        for t in trades:
            t["trade_date"] = trade_dt_str
            t["window"] = label
            t["v2_strike"] = selected_strike
            t["v2_confidence"] = round(v2_result["confidence"], 4)

        day_results.append({
            "window": label,
            "start_mins": start_mins,
            "trade_date": trade_dt_str,
            "v2_strike": selected_strike,
            "v2_confidence": round(v2_result["confidence"], 4),
            "pnl": round(day_pnl, 1),
            "trades": len(trades),
            "trade_details": trades,
        })

    return day_results


# ── Main ──────────────────────────────────────────────────
def run_time_window_backtest(start_date="2025-01-01", end_date="2025-12-31"):
    print(f"\n🔱 Time-Window Sweep Backtest: Krishna V2")
    print(f"   VWAP Threshold : {VWAP_THRESHOLD_PCT}%")
    print(f"   Daily SL       : {DAILY_SL}")
    print(f"   Period         : {start_date} → {end_date}")
    windows_str = ", ".join(w[0] for w in WINDOWS)
    print(f"   Windows        : {windows_str}\n")

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
        print("   Loading market context...")
        feat_df = pd.read_sql(q_feat, conn)
        print("   Loading candles (takes ~2 min)...")
        candles = pd.read_sql(q_cand + " ORDER BY trade_date, strike, ts", conn)

    print(f"   ✅ Data loaded in {timer.time() - t0:.1f}s — {len(candles):,} candles")

    # Timezone + fast lookups
    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"] = candles["ts"].dt.tz_convert(IST)
    candles["_mins"] = candles["ts"].dt.hour * 60 + candles["ts"].dt.minute
    candles["_ts_str"] = candles["ts"].dt.strftime("%H:%M:%S")  # pre-stringify for pickle efficiency

    for col in ["straddle_price", "vwap", "ce_close", "pe_close", "ce_oi", "pe_oi",
                "synthetic_spot", "ce_iv", "pe_iv", "ce_delta", "pe_delta",
                "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", "ce_vega", "pe_vega",
                "vwap_gap_pct", "atm", "ce_volume", "pe_volume"]:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    days = sorted(candles["trade_date"].unique())
    keep_cols = [c for c in [
        "trade_date", "strike", "atm", "straddle_price", "vwap",
        "ce_close", "pe_close", "ce_oi", "pe_oi", "ce_volume", "pe_volume",
        "synthetic_spot", "ce_iv", "pe_iv", "ce_delta", "pe_delta",
        "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", "ce_vega", "pe_vega",
        "vwap_gap_pct", "_mins", "_ts_str", "expiry"
    ] if c in candles.columns]

    pool_args = []
    for trade_date in days:
        date_str = str(trade_date)
        day_data = candles[candles["trade_date"] == trade_date][keep_cols].copy()
        if len(day_data) < 100:
            continue

        # 9:20 baseline for V2
        baseline_920 = {}
        for strike in day_data["strike"].unique():
            sd = day_data[day_data["strike"] == strike]
            snap = sd[sd["_mins"] <= 9 * 60 + 20]
            if snap.empty:
                continue
            r = snap.iloc[-1]
            baseline_920[int(strike)] = {
                "spot": float(r["synthetic_spot"]),
                "straddle": float(r["straddle_price"]),
                "ce_oi": float(r["ce_oi"]),
                "pe_oi": float(r["pe_oi"]),
            }
        if not baseline_920:
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

        pool_args.append((day_data.to_dict("list"), baseline_920, mc, trade_dt_str, expiry_str))

    cores = max(1, os.cpu_count() - 2)
    print(f"\n   🚀 Processing {len(pool_args)} days × {len(WINDOWS)} windows on {cores} cores...")
    t1 = timer.time()

    with mp.Pool(processes=cores) as pool:
        raw = pool.map(_process_day_windows, pool_args)

    print(f"   ✅ Simulation complete in {timer.time() - t1:.1f}s")

    # Flatten results
    all_day_rows = []
    all_trade_rows = []
    for day_list in raw:
        for r in day_list:
            all_trade_rows.extend(r.pop("trade_details", []))
            all_day_rows.append(r)

    if not all_day_rows:
        print("❌ No results.")
        return

    day_df = pd.DataFrame(all_day_rows)

    # ── Window comparison table ──────────────────────────
    print("\n" + "=" * 80)
    print(f"  {'Window':<20} {'Days':>6} {'Trades':>7} {'Win%':>6} {'Total P&L':>11} {'Avg/Day':>9} {'MaxDD':>9} {'Sharpe':>7}")
    print("=" * 80)

    comparison = []
    for (label, start_mins, sel_mins, max_t) in WINDOWS:
        wdf = day_df[day_df["window"] == label]
        if wdf.empty:
            continue
        total = wdf["pnl"].sum()
        avg = wdf["pnl"].mean()
        win_rate = (wdf["pnl"] > 0).mean()
        std = wdf["pnl"].std()
        sharpe = avg / std * (252 ** 0.5) if std > 0 else 0
        cum = wdf["pnl"].cumsum()
        max_dd = (cum - cum.cummax()).min()
        trades = wdf["trades"].sum()

        comparison.append({
            "Window": label,
            "Days": len(wdf),
            "Total Trades": int(trades),
            "Win Rate": f"{win_rate:.0%}",
            "Total P&L (pts)": round(total, 1),
            "Avg Daily P&L": round(avg, 1),
            "Max Drawdown": round(max_dd, 1),
            "Sharpe": round(sharpe, 2),
        })
        print(f"  {label:<20} {len(wdf):>6} {int(trades):>7} {win_rate:>6.0%} {total:>11,.1f} {avg:>9.1f} {max_dd:>9.1f} {sharpe:>7.2f}")

    print("=" * 80 + "\n")

    # ── Save outputs ────────────────────────────────────
    comp_df = pd.DataFrame(comparison)

    # Detailed trades sheet
    trade_df = pd.DataFrame(all_trade_rows)

    # Excel with two sheets
    excel_path = os.path.join(
        config.LOGS_PATH,
        f"v2_time_window_backtest_{start_date}_to_{end_date}.xlsx"
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        comp_df.to_excel(writer, sheet_name="Window Comparison", index=False)
        day_df[["trade_date", "window", "v2_strike", "v2_confidence", "pnl", "trades"]].to_excel(
            writer, sheet_name="Daily by Window", index=False
        )
        if not trade_df.empty:
            cols = ["trade_date", "window", "v2_strike", "v2_confidence",
                    "entry_time", "entry_price", "exit_time", "exit_price", "pnl_pts", "exit_reason"]
            trade_df[[c for c in cols if c in trade_df.columns]].to_excel(
                writer, sheet_name="All Trades", index=False
            )

    print(f"   📊 Excel report → {excel_path}")

    # Also save CSVs
    comp_csv = os.path.join(config.LOGS_PATH, f"v2_window_comparison_{start_date}_to_{end_date}.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"   📄 Comparison    → {comp_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    args = parser.parse_args()
    run_time_window_backtest(args.start, args.end)
