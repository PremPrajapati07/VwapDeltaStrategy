# ============================================================
#  main.py — Entry Point: Run any mode from command line
# ============================================================
"""
Usage:
  python main.py --mode backfill     # Pull historical data from Kite
  python main.py --mode train        # Train ML model on collected data
  python main.py --mode backtest     # Simulate strategy on historical data
  python main.py --mode live         # Run live (paper trade by default)
  python main.py --mode live --real  # Run with real orders (caution!)
"""

import argparse
import datetime as dt
import os
import time
import pandas as pd
from zoneinfo import ZoneInfo


import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)


import config
import logging

# ── Centralized Logging ──────────────────────────────────────
# Initialize immediately before any other local imports to ensure consistency
if not os.path.exists(config.LOGS_PATH):
    os.makedirs(config.LOGS_PATH)

_log_file = os.path.join(config.LOGS_PATH, "trading_engine.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8')
    ]
)
print(f"📝 Logging initialized. File: {_log_file}")

import db
import kite_auth
import data_collector as dc
import krishna_model as ml
import arjun_model as arjun_model
import live_trader_v1 as lt

IST = ZoneInfo("Asia/Kolkata")


# ── Backtest ─────────────────────────────────────────────────

def run_backtest(start_date=None, end_date=None, sl_multiplier=config.SL_MULTIPLIER):
    print(f"\n📈 Running Backtest from {start_date or 'ALL'} to {end_date or 'ALL'} …\n")

    with db.get_conn() as conn:
        q_feat = "SELECT * FROM daily_features"
        q_cand = "SELECT * FROM straddle_candles"
        if start_date or end_date:
            conds = []
            if start_date: conds.append(f"trade_date >= '{start_date}'")
            if end_date:   conds.append(f"trade_date <= '{end_date}'")
            where_clause = " WHERE " + " AND ".join(conds)
            q_feat += where_clause
            q_cand += where_clause

        feat_df = pd.read_sql(q_feat, conn)
        candles = pd.read_sql(q_cand + " ORDER BY trade_date, strike, ts", conn)

    if feat_df.empty:
        print("❌ No feature data found. Run --mode train first.")
        return

    numeric_cols = [
        "ce_iv", "pe_iv", "ce_delta", "pe_delta",
        "ce_theta", "pe_theta", "ce_gamma", "pe_gamma",
        "ce_vega", "pe_vega", "synthetic_spot", "vwap", "vwap_gap_pct",
        "straddle_price", "ce_close", "pe_close", "ce_oi", "pe_oi", "atm"
    ]
    for col in numeric_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"] = candles["ts"].dt.tz_convert(IST)

    print(f"DEBUG: Total candles loaded: {len(candles)}")
    print(f"DEBUG: Unique dates in candles: {candles['trade_date'].unique().tolist()}")

    try:
        model  = ml.load_model()
        use_ml = True
        print("   Using Model Krishna for strike selection.")
    except FileNotFoundError:
        use_ml = False
        print("   ⚠️  No Model Krishna found. Using ATM as fallback.")

    try:
        arjun     = arjun_model.load_arjun_model()
        use_arjun = True
        print("   Using Model Arjun for mid-day exits.\n")
    except FileNotFoundError:
        use_arjun = False
        print("   ⚠️  No Model Arjun found. Using standard SL/VWAP exit.\n")

def _worker_backtest_day_v1(trade_date, day_feat, day_candles, use_ml, model, use_arjun, arjun, sl_multiplier):
    """
    Worker function to process a single day in parallel.
    Everything inside this function must be isolated and picklable.
    """
    if use_ml:
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        if X_rows.empty:
            selected_strike   = int(day_feat.iloc[0]["atm"])
            confidence        = 0.0
            best_feature_dict = {}
        else:
            # XGBoost predict_proba is thread-safe/process-safe
            proba           = model.predict_proba(X_rows)[:, 1]
            best_idx        = proba.argmax()
            selected_strike = int(day_feat.iloc[best_idx]["strike"])
            confidence      = float(proba[best_idx])
            best_feature_dict = X_rows.iloc[best_idx].to_dict()

        best_pnl_idx       = day_feat["pnl"].idxmax()
        best_strike_actual = int(day_feat.loc[best_pnl_idx, "strike"])

        # Log to DB (Each process has its own connection via db.get_conn())
        ml.log_prediction_to_db(
            trade_date=trade_date,
            predicted_strike=selected_strike,
            confidence=confidence,
            features_dict=best_feature_dict,
            actual_best_strike=best_strike_actual,
            prediction_correct=(selected_strike == best_strike_actual)
        )

        if confidence < config.KRISHNA_MIN_CONFIDENCE:
            return {
                "summary": {
                    "trade_date":      trade_date,
                    "selected_strike": selected_strike,
                    "best_strike":     best_strike_actual,
                    "pnl":             0.0,
                    "best_pnl":        day_feat["pnl"].max(),
                    "picked_best":     False,
                    "trades_count":    0,
                    "skip_reason":     f"LOW_CONF({confidence:.0%})"
                },
                "detailed_trades": []
            }
    else:
        selected_strike = int(day_feat.iloc[0]["atm"])
        best_strike_actual = selected_strike # dummy

    strike_candles = day_candles[day_candles["strike"] == selected_strike]
    if strike_candles.empty:
        return None

    strike_candles = strike_candles.rename(columns={"ts": "datetime"})

    day_trades = []
    if use_arjun:
        day_trades = arjun_model.simulate_pnl_with_arjun(
            strike_candles,
            arjun_model=arjun,
            threshold=config.ARJUN_EXIT_THRESHOLD
        )
    else:
        pnl        = ml.simulate_pnl(strike_candles, sl_multiplier=sl_multiplier)
        day_trades = [{
            "entry_time":  strike_candles.iloc[0]["datetime"],
            "entry_price": 0,
            "exit_time":   strike_candles.iloc[-1]["datetime"],
            "exit_price":  0,
            "pnl_pts":     pnl,
            "exit_reason": "STATIC_SL_OR_VWAP"
        }]

    detailed_trades_day = []
    for tr in day_trades:
        tr["trade_date"]      = trade_date
        tr["selected_strike"] = selected_strike
        detailed_trades_day.append(tr)

    day_total_pnl = sum(t["pnl_pts"] for t in day_trades)
    best_pnl      = day_feat["pnl"].max()
    best_strike   = int(day_feat.loc[day_feat["pnl"].idxmax(), "strike"])

    return {
        "summary": {
            "trade_date":      trade_date,
            "selected_strike": selected_strike,
            "best_strike":     best_strike,
            "pnl":             day_total_pnl,
            "best_pnl":        best_pnl,
            "picked_best":     selected_strike == best_strike,
            "trades_count":    len(day_trades)
        },
        "detailed_trades": detailed_trades_day
    }


def run_backtest(start_date=None, end_date=None, sl_multiplier=config.SL_MULTIPLIER):
    print(f"\n📈 Running Parallel Backtest from {start_date or 'ALL'} to {end_date or 'ALL'} …\n")

    with db.get_conn() as conn:
        q_feat = "SELECT * FROM daily_features"
        q_cand = "SELECT * FROM straddle_candles"
        if start_date or end_date:
            conds = []
            if start_date: conds.append(f"trade_date >= '{start_date}'")
            if end_date:   conds.append(f"trade_date <= '{end_date}'")
            where_clause = " WHERE " + " AND ".join(conds)
            q_feat += where_clause
            q_cand += where_clause

        feat_df = pd.read_sql(q_feat, conn)
        candles = pd.read_sql(q_cand + " ORDER BY trade_date, strike, ts", conn)

    if feat_df.empty:
        print("❌ No feature data found. Run --mode train first.")
        return

    numeric_cols = [
        "ce_iv", "pe_iv", "ce_delta", "pe_delta",
        "ce_theta", "pe_theta", "ce_gamma", "pe_gamma",
        "ce_vega", "pe_vega", "synthetic_spot", "vwap", "vwap_gap_pct",
        "straddle_price", "ce_close", "pe_close", "ce_oi", "pe_oi", "atm", "strike"
    ]
    for col in numeric_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    candles["ts"] = pd.to_datetime(candles["ts"])
    if candles["ts"].dt.tz is None:
        candles["ts"] = candles["ts"].dt.tz_localize("UTC")
    candles["ts"] = candles["ts"].dt.tz_convert(IST)

    try:
        model  = ml.load_model()
        use_ml = True
        print("   Using Model Krishna for strike selection.")
    except Exception:
        model  = None
        use_ml = False
        print("   ⚠️  No Model Krishna found. Using ATM as fallback.")

    try:
        arjun     = arjun_model.load_arjun_model()
        use_arjun = True
        print("   Using Model Arjun for mid-day exits.\n")
    except Exception:
        arjun     = None
        use_arjun = False
        print("   ⚠️  No Model Arjun found. Using standard SL/VWAP exit.\n")

    results            = []
    all_detailed_trades = []
    days               = sorted(feat_df["trade_date"].unique())

    # ── Parallel Execution ──
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    tasks = []
    for trade_date in days:
        date_str    = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        day_feat    = feat_df[pd.to_datetime(feat_df["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        day_candles = candles[pd.to_datetime(candles["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        tasks.append((trade_date, day_feat, day_candles, use_ml, model, use_arjun, arjun, sl_multiplier))

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"🚀 Dispatching {len(tasks)} days to {max_workers} processes (Parallel Mode) ...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_backtest_day_v1, *task) for task in tasks]
        
        completed = 0
        for f in futures:
            res = f.result()
            completed += 1
            if res:
                results.append(res["summary"])
                all_detailed_trades.extend(res["detailed_trades"])
            
            if completed % 10 == 0 or completed == len(tasks):
                print(f"   Progress: {completed}/{len(tasks)} days processed ...")

    if not results:
        print("No backtest results.")
        return

    res_df        = pd.DataFrame(results)
    total_pnl     = res_df["pnl"].sum()
    best_possible = res_df["best_pnl"].sum()
    win_days      = (res_df["pnl"] > 0).sum()
    loss_days     = (res_df["pnl"] <= 0).sum()
    accuracy      = res_df["picked_best"].mean()
    avg_pnl       = res_df["pnl"].mean()
    max_dd        = res_df["pnl"].cumsum().sub(res_df["pnl"].cumsum().cummax()).min()
    sharpe        = (res_df["pnl"].mean() / res_df["pnl"].std() * (252 ** 0.5)
                     if res_df["pnl"].std() > 0 else 0)

    print("=" * 60)
    print("  BACKTEST RESULTS (VWAP Straddle Strategy)")
    print("=" * 60)
    print(f"  Period          : {res_df['trade_date'].min()} → {res_df['trade_date'].max()}")
    print(f"  Trading Days    : {len(res_df)}")
    print(f"  Win Days        : {win_days}  ({win_days/len(res_df):.0%})")
    print(f"  Loss Days       : {loss_days}")
    print(f"  Total P&L (pts) : {total_pnl:,.1f}")
    print(f"  Best Possible   : {best_possible:,.1f}")
    if best_possible:
        print(f"  Capture Rate    : {total_pnl/best_possible:.0%}")
    print(f"  Avg Daily P&L   : {avg_pnl:.1f} pts")
    print(f"  Max Drawdown    : {max_dd:.1f} pts")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Strike Accuracy : {accuracy:.0%}  (ML picked best strike)")
    print("=" * 60)

    report_path = os.path.join(config.LOGS_PATH, "backtest_report.csv")
    res_df.to_csv(report_path, index=False)

    detailed_path = os.path.join(config.LOGS_PATH, "backtest_trades_detailed.csv")
    detailed_df   = pd.DataFrame(all_detailed_trades)
    if not detailed_df.empty:
        cols        = ["trade_date", "selected_strike", "entry_time", "entry_price",
                       "exit_time", "exit_price", "pnl_pts", "exit_reason"]
        detailed_df = detailed_df[[c for c in cols if c in detailed_df.columns]]
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  Detailed trade log saved → {detailed_path}")

    print(f"\n  Daily summary saved → {report_path}\n")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nifty VWAP Straddle Strategy")
    parser.add_argument("--mode", choices=["backfill", "train", "backtest", "live"],
                        required=True)
    parser.add_argument("--real",           action="store_true")
    parser.add_argument("--lots",           type=int,   default=1)
    parser.add_argument("--start",          default=config.BACKTEST_START)
    parser.add_argument("--end",            default=config.BACKTEST_END)
    parser.add_argument("--sl_multiplier",  type=float, default=config.SL_MULTIPLIER)
    parser.add_argument("--skip-kite",      action="store_true")
    parser.add_argument("--skip-neo",       action="store_true")
    args = parser.parse_args()

    os.makedirs(config.LOGS_PATH,   exist_ok=True)
    os.makedirs(config.MODELS_PATH, exist_ok=True)

    os.makedirs(config.LOGS_PATH, exist_ok=True)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_PATH, "trading_engine.log"))
        ]
    )
    
    db.init_schema()

    # ✅ FIX: init Kite when enabled (needed for market data even in Neo mode)
    kite = None
    if config.ENABLE_ZERODHA and not args.skip_kite:
        if args.mode in ("backfill", "train", "live"):
            print("Ensuring Kite access token ...")
            kite = kite_auth.get_kite()

    if args.mode == "backfill":
        if not kite:
            print("❌ Backfill requires Zerodha Kite. Enable it in .env (ENABLE_ZERODHA=True).")
            return
        dc.backfill_history(kite, start_date=args.start, end_date=args.end)
        print("\nBackfill complete. Run --mode train next.\n")

    elif args.mode == "train":
        print("\nTraining ML model ...\n")
        feat_df = ml.build_features_from_db()
        ml.train_model(feat_df)
        print("\nModel trained. Run --mode backtest to validate.\n")

    elif args.mode == "backtest":
        run_backtest(args.start, args.end, sl_multiplier=args.sl_multiplier)

    elif args.mode == "live":
        paper = not args.real
        if not paper:
            confirm = input("!!! REAL ORDER MODE. Type 'YES' to confirm: ")
            if confirm.strip() != "YES":
                print("Aborted.")
                return

        if config.DEBUG_FORCE_TEST_ORDER:
            print("⚠️  DEBUG_FORCE_TEST_ORDER=True — skipping all market-time checks!")
            print("   → Orders will fire immediately regardless of time.")
        elif config.WAIT_FOR_MARKET_OPEN:
            now_ist   = dt.datetime.now(IST)
            today     = now_ist.date()
            start_ist = dt.datetime.combine(today, dt.time(9, 15), tzinfo=IST)
            end_ist   = dt.datetime.combine(today, dt.time(15, 30), tzinfo=IST)

            if now_ist > end_ist:
                print("🛑 Market already closed for today. Exiting.")
                return

            if now_ist < start_ist:
                wait_sec = (start_ist - now_ist).total_seconds()
                print(f"🕙 Market opens at 9:15 AM IST. "
                      f"Current IST: {now_ist.strftime('%H:%M:%S')}. "
                      f"Waiting {wait_sec/60:.1f} minutes …")
                time.sleep(wait_sec)
            else:
                print(f"✅ Market already open (IST: {now_ist.strftime('%H:%M:%S')}). Starting immediately …")


        lt.run_live_trading(
            lots=args.lots,
            paper_trade=paper,
            skip_kite=args.skip_kite,
            skip_neo=args.skip_neo
        )


if __name__ == "__main__":
    main()
