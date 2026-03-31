# ============================================================
#  scripts/live_simulation.py — Live Strategy Replay Engine
# ============================================================
"""
Simulate a "Live" trading day using historical database candles.
This script reproduces the exact behavior of live_trader.py:
  1. 9:15 — Data loading
  2. 9:20 — Krishna selects strike
  3. 9:20 onwards — Arjun monitors exit per minute
"""

import os
import sys
import time
import argparse
import datetime as dt
import pandas as pd
import logging

# Add parent dir to path
sys.path.append(os.getcwd())

import config
import db
import krishna_model as ml
import arjun_model

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_simulation(target_date: str, speed: float = 0.05):
    """
    Replay a specific date minute-by-min.
    speed: seconds to wait between "minutes" (default 0.05s for fast replay)
    """
    print(f"\n{'='*60}")
    print(f" 🎬  STARTING LIVE SIMULATION — {target_date}")
    print(f"      Models: Krishna (Strategist) & Arjun (Warrior)")
    print(f"{'='*60}\n")

    # 1. Load data from DB
    with db.get_conn() as conn:
        # Load all candles for the day
        candles = pd.read_sql(f"""
            SELECT * FROM straddle_candles 
            WHERE trade_date = '{target_date}'
            ORDER BY ts, strike
        """, conn)
        
        # Load market context
        ctx = pd.read_sql(f"SELECT * FROM market_context WHERE trade_date = '{target_date}'", conn)
        
        # Load daily features (for Krishna selection at 9:20)
        feats = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date = '{target_date}'", conn)

    if candles.empty or ctx.empty or feats.empty:
        print(f"❌ Error: Missing data for {target_date}. Ensure backfill and train are complete.")
        return

    # Numeric conversions
    numeric_cols = ["straddle_price", "vwap", "ce_volume", "pe_volume", "ce_iv", "pe_iv", 
                    "ce_delta", "pe_delta", "ce_theta", "pe_theta"]
    for col in numeric_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)

    # 2. Setup Models
    try:
        krishna = ml.load_model()
        arjun = arjun_model.load_arjun_model()
        print("✅ Models Krishna & Arjun loaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # 3. Simulation Loop
    times = sorted(candles["ts"].unique())
    expiry = ctx.iloc[0]["expiry"]
    vix = float(ctx.iloc[0]["vix"])
    nifty_prev_change = float(ctx.iloc[0]["nifty_prev_day_change"])
    nifty_open_gap = float(ctx.iloc[0]["nifty_open_gap"])

    position = None
    selected_strike = None
    total_pnl = 0.0
    
    # Trackers
    vol_window = []
    
    print("⏳ Replaying market timeline ...")
    
    for ts in times:
        ts_ist = pd.to_datetime(ts).tz_convert("Asia/Kolkata")
        t_str = ts_ist.strftime("%H:%M")
        
        # --- 9:20 AM: Krishna Selection ---
        if t_str == "09:20":
            # Filter feats for this date
            X_rows = feats[config.ML_FEATURES].fillna(0)
            proba = krishna.predict_proba(X_rows)[:, 1]
            best_idx = proba.argmax()
            selected_strike = int(feats.iloc[best_idx]["strike"])
            confidence = proba[best_idx]
            
            print(f"🤖 09:20 | Krishna selects Strike: {selected_strike} (Confidence: {confidence:.1%})")
            
            # Log prediction to DB (mocking the live call)
            best_feat_dict = X_rows.iloc[best_idx].to_dict()
            ml.log_prediction_to_db(target_date, selected_strike, confidence, best_feat_dict)

        if not selected_strike:
            continue

        # Get current candle for selected strike
        row = candles[(candles["ts"] == ts) & (candles["strike"] == selected_strike)]
        if row.empty: continue
        row = row.iloc[0]
        
        price = float(row["straddle_price"])
        vwap  = float(row["vwap"])
        vol   = float(row["ce_volume"] + row["pe_volume"])
        gap_pct = (price - vwap) / vwap * 100 if vwap else 0

        # --- Position Logic ---
        if position is None:
            # Entry: Price below VWAP after 9:20
            if price < vwap and ts_ist.time() >= dt.time(9, 20):
                position = {
                    "strike": selected_strike,
                    "entry_price": price,
                    "max_pnl": 0.0,
                    "entry_iv": float(row["ce_iv"] + row["pe_iv"]),
                    "entry_time": ts_ist
                }
                print(f"🟢 {t_str} | ENTRY @ {price:.1f} | Strike {selected_strike}")
        else:
            # Monitoring Open Position
            pnl_pts = position["entry_price"] - price
            position["max_pnl"] = max(position["max_pnl"], pnl_pts)
            drawdown = position["max_pnl"] - pnl_pts
            
            # Arjun Features
            vol_window.append(vol)
            if len(vol_window) > 15: vol_window.pop(0)
            avg_vol = max(sum(vol_window) / len(vol_window), 1.0)
            rel_vol = min(vol / avg_vol, 100.0)
            
            curr_iv = float(row["ce_iv"] + row["pe_iv"])
            iv_d = curr_iv - position["entry_iv"]
            delta_d = abs(float(row["ce_delta"] + row["pe_delta"]))
            theta_v = float(row["ce_theta"] + row["pe_theta"])

            # Call Arjun
            arjun_res = arjun_model.predict_exit(
                pnl_pts=pnl_pts, max_pnl_so_far=position["max_pnl"], drawdown=drawdown,
                vwap_gap_pct=gap_pct, delta_drift=delta_d, theta_velocity=theta_v,
                iv_drift=iv_d, rel_vol_15m=rel_vol, model=arjun,
                threshold=config.ARJUN_EXIT_THRESHOLD
            )

            # --- Status Print ---
            print(f"  {t_str} | Price: {price:.1f} | VWAP: {vwap:.1f} | P&L: {pnl_pts:+.1f} | Arjun: {arjun_res['confidence']:.0%}" + 
                  (" ⚠️ EXIT SIGNAL" if arjun_res["should_exit"] else ""))

            # Exit Conditions: Arjun OR VWAP crossover OR Hard Exit
            exit_reason = None
            if arjun_res["should_exit"]: exit_reason = f"ARJUN_EXIT ({arjun_res['confidence']:.0%})"
            elif price > vwap:           exit_reason = "VWAP_CROSS"
            elif ts_ist.time() >= dt.time(15, 0): exit_reason = "HARD_EXIT"

            if exit_reason:
                final_pnl = pnl_pts * config.LOT_SIZE
                total_pnl += final_pnl
                print(f"🔴 {t_str} | EXIT @ {price:.1f} | Reason: {exit_reason} | P&L: ₹{final_pnl:,.0f}")
                
                # Log to DB
                with db.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO trade_log (trade_date, expiry, selected_strike, atm, entry_time, entry_premium, exit_time, exit_premium, exit_reason, pnl, lots)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
                        """, (target_date, expiry, selected_strike, selected_strike, position["entry_time"], position["entry_price"], ts_ist, price, exit_reason, final_pnl))
                
                position = None
                # Stop simulation after first exit for brevity or continue?
                # Usually we can look for re-entry, but let's exit for today in simulation header
                # break 

        time.sleep(speed)

    print(f"\n{'='*60}")
    print(f" 🏁 SIMULATION COMPLETE")
    print(f" Total Net P&L: ₹{total_pnl:,.0f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-03-30", help="Date to simulate (YYYY-MM-DD)")
    parser.add_argument("--speed", type=float, default=0.01, help="Seconds between minutes")
    args = parser.parse_args()
    
    run_simulation(args.date, args.speed)
