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

First-time setup:
  1. Copy .env.example to .env and fill in all values
  2. Ensure PostgreSQL is running and DATABASE_URL is set
  3. Run: python main.py --mode backfill
  4. Then: train → backtest → live

Login is fully automated — no manual token needed!
"""

import argparse
import datetime as dt
import os
import pandas as pd

import config
import db
import kite_auth
import data_collector as dc
import krishna_model as ml
import arjun_model
import live_trader as lt


# ── Backtest ─────────────────────────────────────────────────

def run_backtest(start_date=None, end_date=None, sl_multiplier=config.SL_MULTIPLIER):
    """
    Full backtest: for each historical Thursday, simulate the strategy
    using VWAP crossover on the strike the ML model would have chosen.
    Prints a detailed P&L report.
    """
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
        print("❌ No feature data found for these dates. Run --mode train first.")
        return

    # Ensure numeric types for ALL price and calc columns (Postgres may return Decimals as objects)
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
    
    print(f"DEBUG: Total candles loaded: {len(candles)}")
    print(f"DEBUG: Unique dates in candles: {candles['trade_date'].unique().tolist()}")

    try:
        model = ml.load_model()
        use_ml = True
        print("   Using Model Krishna for strike selection.")
    except FileNotFoundError:
        use_ml = False
        print("   ⚠️  No Model Krishna found. Using ATM as fallback.")

    try:
        arjun = arjun_model.load_arjun_model()
        use_arjun = True
        print("   Using Model Arjun for mid-day exits.\n")
    except FileNotFoundError:
        use_arjun = False
        print("   ⚠️  No Model Arjun found. Using standard ST/VWAP exit.\n")

    results = []
    days = sorted(feat_df["trade_date"].unique())

    for trade_date in days:
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        day_feat = feat_df[pd.to_datetime(feat_df["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        day_candles = candles[pd.to_datetime(candles["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]

        if use_ml:
            X_rows = day_feat[config.ML_FEATURES].fillna(0)
            if X_rows.empty:
                selected_strike = int(day_feat.iloc[0]["atm"])
                confidence = 0.0
                best_feature_dict = {}
            else:
                proba = model.predict_proba(X_rows)[:, 1]
                best_idx = proba.argmax()
                selected_strike = int(day_feat.iloc[best_idx]["strike"])
                confidence = float(proba[best_idx])
                best_feature_dict = X_rows.iloc[best_idx].to_dict()
            
            # --- LOG TO DATABASE (un-empty the table) ---
            best_pnl_idx = day_feat["pnl"].idxmax()
            best_strike_actual = int(day_feat.loc[best_pnl_idx, "strike"])
            
            ml.log_prediction_to_db(
                trade_date=trade_date,
                predicted_strike=selected_strike,
                confidence=confidence,
                features_dict=best_feature_dict,
                actual_best_strike=best_strike_actual,
                prediction_correct=(selected_strike == best_strike_actual)
            )
        else:
            selected_strike = int(day_feat.iloc[0]["atm"])

        strike_candles = day_candles[day_candles["strike"] == selected_strike]
        if strike_candles.empty:
            continue

        # Rename ts → datetime for simulate_pnl compatibility
        strike_candles = strike_candles.rename(columns={"ts": "datetime"})
        
        if use_arjun:
            pnl, reason, t_exit = arjun_model.simulate_pnl_with_arjun(
                strike_candles, 
                arjun_model=arjun, 
                threshold=config.ARJUN_EXIT_THRESHOLD
            )
        else:
            # Fallback to standard ML/VWAP simulator
            pnl = ml.simulate_pnl(strike_candles, sl_multiplier=sl_multiplier)
            reason = "STATIC_SL_OR_VWAP"

        best_pnl    = day_feat["pnl"].max()
        best_strike = int(day_feat.loc[day_feat["pnl"].idxmax(), "strike"])

        results.append({
            "trade_date":      trade_date,
            "selected_strike": selected_strike,
            "best_strike":     best_strike,
            "pnl":             pnl,
            "best_pnl":        best_pnl,
            "picked_best":     selected_strike == best_strike,
            "exit_reason":     reason
        })

    if not results:
        print("No backtest results.")
        return

    res_df = pd.DataFrame(results)

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
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    res_df.to_csv(report_path, index=False)
    print(f"\n  Full report saved → {report_path}\n")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nifty VWAP Straddle Strategy")
    parser.add_argument("--mode", choices=["backfill", "train", "backtest", "live"],
                        required=True, help="Which mode to run")
    parser.add_argument("--real",  action="store_true",
                        help="Place real orders (default is paper trade)")
    parser.add_argument("--lots",  type=int, default=1,
                        help="Number of lots to trade (default: 1)")
    parser.add_argument("--start", default=config.BACKTEST_START,
                        help="Backfill start date YYYY-MM-DD")
    parser.add_argument("--end",   default=config.BACKTEST_END,
                        help="Backfill end date YYYY-MM-DD")
    parser.add_argument("--sl_multiplier", type=float, default=config.SL_MULTIPLIER,
                        help="Stop Loss multiplier (default from config)")
    args = parser.parse_args()

    # ── Startup: init schema + auto-login ──
    os.makedirs(config.LOGS_PATH,   exist_ok=True)
    os.makedirs(config.MODELS_PATH, exist_ok=True)

    print("Initializing PostgreSQL schema ...")
    db.init_schema()

    print("Ensuring Kite access token ...")
    kite = kite_auth.get_kite()

    # ── Route to mode ──
    if args.mode == "backfill":
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
            if confirm != "YES":
                print("Aborted.")
                return
        
        # ── Automated Scheduler ──
        if config.WAIT_FOR_MARKET_OPEN:
            now = dt.datetime.now()
            start_time = dt.datetime.combine(now.date(), dt.time(9, 15))
            end_time = dt.datetime.combine(now.date(), dt.time(15, 30))
            
            if now < start_time:
                wait_sec = (start_time - now).total_seconds()
                print(f"🕙 Market not open yet. Waiting {wait_sec/60:.1f} minutes until 9:15 AM …")
                time.sleep(wait_sec)
            elif now > end_time:
                print(f"🛑 Market already closed for today. exiting.")
                return

        lt.run_live_trading(kite, lots=args.lots, paper_trade=paper)


if __name__ == "__main__":
    import time
    main()
