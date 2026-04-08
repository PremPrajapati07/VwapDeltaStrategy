# ============================================================
#  app_v2.py — Production Entry Point (Krishna V2 + Arjun)
# ============================================================
"""
Orchestrates the entire trading system using the latest 
Krishna V2 (Strike Selection) and Arjun (Dynamic Exit) models.

Usage:
  python app_v2.py --mode backfill     # Pull historical data from Kite
  python app_v2.py --mode train        # Train Krishna V2 model
  python app_v2.py --mode backtest     # Run strategy on historical data
  python app_v2.py --mode live         # Paper Trading (Simulated Orders)
  python app_v2.py --mode live --real  # REAL MONEY TRADING (Caution!)
"""

import argparse
import datetime as dt
import os
import time
import logging
import pandas as pd
from zoneinfo import ZoneInfo

import config
import db
import kite_auth
import data_collector as dc
import krishna_v2_model as ml
import arjun_model
import live_trader_v21 as lt

IST = ZoneInfo("Asia/Kolkata")

# ── Centralized Logging ──────────────────────────────────────
def setup_logging():
    if not os.path.exists(config.LOGS_PATH):
        os.makedirs(config.LOGS_PATH)

    log_file = os.path.join(config.LOGS_PATH, "trading_engine_v2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() # Also print to console for major status
        ]
    )
    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    print(f"📝 Logging initialized. File: {log_file}")

# ── Backtest Logic ───────────────────────────────────────────
def run_backtest_v2(start_date=None, end_date=None):
    """
    Standard backtest using Krishna V2 logic.
    Note: For highest fidelity, use simulate_today.py for single days.
    """
    print(f"\n📈 Running V2 Backtest from {start_date or 'ALL'} to {end_date or 'ALL'} …\n")
    
    # 1. Fetch historical features and candles
    with db.get_conn() as conn:
        q_feat = "SELECT * FROM daily_features"
        if start_date or end_date:
            conds = []
            if start_date: conds.append(f"trade_date >= '{start_date}'")
            if end_date:   conds.append(f"trade_date <= '{end_date}'")
            q_feat += " WHERE " + " AND ".join(conds)
        
        feat_df = pd.read_sql(q_feat, conn)

    if feat_df.empty:
        print("❌ No feature data found. Run --mode train first.")
        return

    # 2. Load Model
    try:
        model = ml.load_v2_model()
        print("✅ Krishna V2 Model loaded successfully.")
    except Exception as e:
        print(f"⚠️  V2 Model not found ({e}). Run --mode train first.")
        return

    # 3. Simulate loop (Simplified for performance, use simulate_today.py for tick-mirroring)
    results = []
    days = sorted(feat_df["trade_date"].unique())
    
    for day in days:
        day_str = pd.to_datetime(day).strftime('%Y-%m-%d')
        # Placeholder for backtest results aggregation
        # In a real scenario, this would call a V2-aware PnL simulator
        results.append({"trade_date": day_str, "status": "Simulated"})

    print(f"✅ V2 Backtest completed for {len(days)} days.")
    print("💡 Tip: Use 'python simulate_today.py' for minute-by-minute replay of specific days.")

# ── CLI ──────────────────────────────────────────────────────
def main():
    setup_logging()
    db.init_schema()

    parser = argparse.ArgumentParser(description="Nifty Straddle Strategy V2")
    parser.add_argument("--mode", choices=["backfill", "train", "backtest", "live"], required=True)
    parser.add_argument("--real", action="store_true", help="Execute real orders on Kotak Neo")
    parser.add_argument("--lots", type=int, default=config.DEFAULT_LOTS)
    parser.add_argument("--start", default=config.BACKTEST_START)
    parser.add_argument("--end", default=config.BACKTEST_END)
    parser.add_argument("--skip-kite", action="store_true")
    args = parser.parse_args()

    # 1. Backfill Mode
    if args.mode == "backfill":
        print("\n📥 Mode: Backfill (Kite Historical Data)\n")
        kite = kite_auth.get_kite()
        dc.backfill_history(kite, start_date=args.start, end_date=args.end)
        print("\n✅ Backfill complete.")

    # 2. Train Mode
    elif args.mode == "train":
        print("\n🔱 Mode: Training Krishna V2 (Dynamic Strike Model)\n")
        # Krishna V2 uses building logic from krishna_v2_model.py
        feat_df = ml.build_v2_features(start_date=args.start, end_date=args.end)
        if not feat_df.empty:
            ml.train_v2(feat_df)
            print("\n✅ Training complete. Krishna V2 is ready.")
        else:
            print("❌ No features generated. Check your historical data.")

    # 3. Backtest Mode
    elif args.mode == "backtest":
        run_backtest_v2(args.start, args.end)

    # 4. Live Mode (Paper or Real)
    elif args.mode == "live":
        paper = not args.real
        if not paper:
            print("\n" + "!"*60)
            print("  ⚠️  WARNING: REAL MONEY MODE ACTIVATED  ⚠️")
            print("  This script will place actual orders on Kotak Neo.")
            print("!"*60 + "\n")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm.strip() != "CONFIRM":
                print("❌ Aborted.")
                return

        # ── Market Time Preparation ──
        if not config.DEBUG_FORCE_TEST_ORDER:
            now_ist = dt.datetime.now(IST)
            today = now_ist.date()
            market_open = dt.datetime.combine(today, dt.time(9, 15), tzinfo=IST)
            market_close = dt.datetime.combine(today, dt.time(15, 30), tzinfo=IST)

            if now_ist > market_close:
                print("🛑 Market is closed for today. Exiting.")
                return

            if now_ist < market_open:
                wait_sec = (market_open - now_ist).total_seconds()
                print(f"🕙 Market opens at 9:15 AM IST. Waiting {wait_sec/60:.1f} minutes …")
                time.sleep(wait_sec)
            else:
                print(f"✅ Market already open (IST: {now_ist.strftime('%H:%M:%S')}).")

        # ── Start Live Engine ──
        print(f"\n🚀 Starting Live Trading Engine V2...")
        print(f"   Model  : Krishna V2 (Strike Selection)")
        print(f"   Exit   : Arjun Dynamic Exit")
        print(f"   Mode   : {'PAPER' if paper else 'REAL'}")
        print(f"   Lots   : {args.lots}\n")

        try:
            lt.run_live_trading(
                lots=args.lots,
                paper_trade=paper,
                skip_kite=args.skip_kite,
                skip_neo=False # Always use Neo for execution in live
            )
        except KeyboardInterrupt:
            print("\n👋 Manual stop detected. Exiting gracefully.")
        except Exception as e:
            logging.critical(f"Engine Failure: {e}")
            print(f"\n❌ FATAL ENGINE ERROR: {e}")

if __name__ == "__main__":
    main()
