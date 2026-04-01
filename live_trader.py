# ============================================================
#  live_trader.py — Live Trading Engine
# ============================================================
"""
Responsibilities:
  1. At 9:15 IST → Connect Kite, start VWAP tracker for all 11 strikes
  2. At 9:20 IST → Run ML model → select best strike
  3. Every minute → check VWAP crossover → entry / exit / re-entry
  4. Re-entry → re-evaluate strike via ML
  5. At 3:00 PM IST → force close all positions
  6. Log every trade to PostgreSQL trade_log table
"""

import os
import json
import time
import logging
import datetime as dt
import pandas as pd
from collections import defaultdict
from zoneinfo import ZoneInfo          # ✅ FIX: IST-aware time throughout

import config
import data_collector as dc
import krishna_model as ml
import arjun_model
import db
import kite_auth
import kotak_auth
from brokers import KiteBroker, NeoBroker

IST = ZoneInfo("Asia/Kolkata")         # ✅ single source of truth for timezone

log = logging.getLogger(__name__)


def _now_ist() -> dt.datetime:
    """Always returns timezone-aware datetime in IST."""
    return dt.datetime.now(IST)


# ── VWAP Tracker (in-memory, updates every tick) ─────────────

class VWAPTracker:
    """Tracks cumulative VWAP for multiple strikes in real time."""

    def __init__(self):
        self._cum_pv  = defaultdict(float)
        self._cum_vol = defaultdict(float)
        self._vwap    = defaultdict(float)

    def update(self, strike: int, price: float, volume: float):
        if volume <= 0:
            return
        self._cum_pv[strike]  += price * volume
        self._cum_vol[strike] += volume
        self._vwap[strike]     = self._cum_pv[strike] / self._cum_vol[strike]

    def get_vwap(self, strike: int) -> float:
        return self._vwap.get(strike, 0.0)

    def get_all(self) -> dict:
        return dict(self._vwap)


# ── Position Manager ─────────────────────────────────────────

class Position:
    def __init__(self, strike: int, ce_symbol: str, pe_symbol: str,
                 entry_premium: float, entry_time: dt.datetime, expiry: dt.date,
                 lots: int = 1, lot_size: int = None):
        self.strike         = strike
        self.expiry         = expiry
        self.ce_symbol      = ce_symbol
        self.pe_symbol      = pe_symbol
        self.entry_premium  = entry_premium
        self.entry_time     = entry_time
        self.lots           = lots
        self.lot_size       = lot_size or config.LOT_SIZE
        self.sl_level       = entry_premium * config.SL_MULTIPLIER
        self.re_entry_count = 0
        self.is_open        = False # Set to True only on successful placement

    def pnl_points(self, current_premium: float) -> float:
        return (self.entry_premium - current_premium) * self.lots * self.lot_size

    def sl_hit(self, current_premium: float) -> bool:
        return current_premium >= self.sl_level


# ── Order Execution ──────────────────────────────────────────

def place_sell_straddle(brokers, position: Position):
    """Place sell orders for CE and PE legs. Returns True ONLY if both succeed on all brokers."""
    if not brokers: return False
    
    all_success = True
    for broker in brokers:
        try:
            log.info(f"[{broker.name}] Placing SELL straddle for {position.strike}")
            ce_id = broker.place_order(
                symbol=position.ce_symbol,
                quantity=position.lots * position.lot_size,
                side="SELL",
                strike=position.strike,
                expiry=position.expiry
            )
            pe_id = broker.place_order(
                symbol=position.pe_symbol,
                quantity=position.lots * position.lot_size,
                side="SELL",
                strike=position.strike,
                expiry=position.expiry
            )
            
            if ce_id and pe_id:
                print(f"  ✅ [{broker.name}] SOLD Strike {position.strike}")
            else:
                all_success = False
                log.error(f"[{broker.name}] Straddle leg failed. CE_ID={ce_id}, PE_ID={pe_id}")
                # Optional: handle partial fills / exits if one leg failed
        except Exception as e:
            all_success = False
            log.error(f"[{broker.name}] Order placement exception: {e}")
            print(f"  ❌ [{broker.name}] Placement EXCEPTION: {e}")

    return all_success


def _exit_position(brokers, position: Position, current_premium: float,
                   reason: str, paper_trade: bool, now: dt.datetime) -> float:
    pnl = position.pnl_points(current_premium)
    position.is_open = False
    if paper_trade:
        print(f"  📝 [PAPER] BUY close straddle {position.strike} @ {current_premium:.1f} | "
              f"Reason: {reason} | P&L: ₹{pnl:,.0f}")
    else:
        for broker in brokers:
            log.info(f"[{broker.name}] Closing position for {position.strike} | reason={reason}")
            broker.place_order(position.ce_symbol, position.lots * position.lot_size, "BUY", 
                               strike=position.strike, expiry=position.expiry)
            broker.place_order(position.pe_symbol, position.lots * position.lot_size, "BUY", 
                               strike=position.strike, expiry=position.expiry)
            print(f"  🔁 [{broker.name}] CLOSED Strike {position.strike} | P&L: ₹{pnl:,.0f}")
    return pnl


def log_trade_to_db(position: Position, exit_time: dt.datetime, exit_premium: float,
                    exit_reason: str, pnl: float, expiry: dt.date, paper_trade: bool = True):
    """Save completed trade to PostgreSQL trade_log table."""
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_log
                (trade_date, expiry, selected_strike, atm, entry_time, entry_premium,
                 exit_time, exit_premium, exit_reason, pnl, lots, re_entry_count, paper_trade)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                position.entry_time.date(),
                expiry,
                int(position.strike),
                int(position.strike),
                position.entry_time,
                float(position.entry_premium),
                exit_time,
                float(exit_premium),
                exit_reason,
                float(pnl),
                int(position.lots),
                int(position.re_entry_count),
                bool(paper_trade),
            ))
    log.info(f"Trade logged: strike={position.strike} pnl={pnl:.0f} reason={exit_reason}")


# ── Risk Helpers ─────────────────────────────────────────────

def check_margins(brokers, lots: int = 1) -> bool:
    """Check if all active brokers have sufficient margin."""
    if not config.CHECK_MARGIN:
        return True

    all_ok = True
    required = config.MIN_REQUIRED_MARGIN * lots
    print(f"\n💰 Checking Margins for {lots} Lot(s) (Required: ₹{required:,.0f} per broker)...")

    for broker in brokers:
        try:
            available = broker.get_balance()
            log.info(f"[{broker.name}] Available: ₹{available:,.0f} | Required: ₹{required:,.0f}")

            if available < required:
                print(f"  ❌ [{broker.name}] INSUFFICIENT FUNDS: ₹{available:,.0f}")
                all_ok = False
            else:
                print(f"  ✅ [{broker.name}] MARGIN OK: ₹{available:,.0f}")
        except Exception as e:
            log.error(f"Margin check failed for {broker.name}: {e}")
            all_ok = False

    if not all_ok:
        print("⚠️  Risk Warning: One or more brokers have insufficient margin. Skipping trade.")

    return all_ok


def load_trading_state():
    """Loads the trading state if it exists and is for the current day."""
    if not os.path.exists(config.STATE_FILE):
        return None
        
    try:
        with open(config.STATE_FILE, "r") as f:
            state = json.load(f)
            
        today_str = dt.datetime.now(IST).strftime("%Y-%m-%d")
        if state.get("date") == today_str:
            return state
    except Exception as e:
        log.error(f"Failed to load trading state: {e}")
        
    return None


def save_trading_state(strike, cum_pv, cum_vol):
    """Saves the current trading progress to disk using VWAP tracker metrics."""
    try:
        state = {
            "date": dt.datetime.now(IST).strftime("%Y-%m-%d"),
            "strike": strike,
            "cum_pv": float(cum_pv),
            "cum_vol": float(cum_vol),
            "updated_at": dt.datetime.now(IST).strftime("%H:%M:%S")
        }
        with open(config.STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        log.error(f"Failed to save trading state: {e}")


def clear_trading_state():
    """Deletes the state file to force a fresh restart."""
    if os.path.exists(config.STATE_FILE):
        try:
            os.remove(config.STATE_FILE)
            log.info("Trading state cleared.")
        except Exception as e:
            log.error(f"Failed to clear state: {e}")


# ── Main Trading Loop ────────────────────────────────────────

def run_live_trading(lots: int = 1, paper_trade: bool = True, skip_kite: bool = False, skip_neo: bool = False):
    """
    Main entry point for live trading day.
    Automatically initializes all enabled brokers, respecting skip overrides.
    All times are IST (Asia/Kolkata).
    """
    today          = _now_ist().date()                 # ✅ IST date
    active_brokers = []
    data_kite      = None

    # ── Phase 0: Initialize Brokers ──────────────────────────
    import kite_auth
    import kotak_auth
    
    # Intelligence: ALWAYS Zerodha (most stable for Greeks/Spot)
    # Execution: Kotak (default for live) or both
    data_broker    = None
    kite_inst      = None
    neo_inst       = None
    active_brokers = []

    # 1. Initialize Zerodha for Intelligence (Data)
    # We always need this for Greeks calculation and ML confidence.
    try:
        kite_inst = kite_auth.get_kite()
        kite_broker = KiteBroker(kite_inst)
        data_broker = kite_broker # Intelligence source
        log.info("Zerodha Kite connected for Market Intelligence (Snapshots/Greeks).")
    except Exception as e:
        log.error(f"Failed to connect Zerodha Kite: {e}")
        print("❌ CRITICAL: Zerodha Kite required for market data. Check credentials.")
        return

    # 2. Initialize Brokers for Execution (Orders)
    if not paper_trade:
        # For real trades, ONLY use Kotak Neo as requested. NO ZERODHA ORDERS.
        if config.ENABLE_KOTAK and not skip_neo:
            try:
                neo_inst = kotak_auth.get_neo_client()
                if neo_inst:
                    active_brokers = [NeoBroker(neo_inst)]
                    log.info("Execution Port: Kotak Neo (Exclusive).")
                else:
                    log.error("Failed to connect Kotak Neo. No execution broker available.")
                    return
            except Exception as e:
                log.error(f"Failed to connect Kotak Neo: {e}")
                print(f"❌ CRITICAL ERROR: Kotak Neo execution failure: {e}")
                return
        else:
            print("❌ ERROR: Real trading mode requires Kotak Neo enabled.")
            return
    else:
        # Paper trade mode: active_brokers can remain empty
        log.info("Running in PAPER TRADE mode. No real broker orders will be placed.")

    if not paper_trade and not active_brokers:
        print("❌ ERROR: No execution brokers available for real trade mode.")
        return

    expiry = dc.get_nearest_expiry(data_broker)

    # ✅ FIX: parse config times as IST-aware datetimes for correct comparison
    hard_exit_h, hard_exit_m = [int(x) for x in config.HARD_EXIT_TIME.split(":")]
    scan_h,      scan_m      = [int(x) for x in config.SCAN_TIME.split(":")]
    open_h,      open_m      = 9, 15

    hard_exit_time = dt.time(hard_exit_h, hard_exit_m)
    scan_time_t    = dt.time(scan_h, scan_m)
    open_time_t    = dt.time(open_h, open_m)

    vwap_tracker         = VWAPTracker()
    position             = None
    total_pnl            = 0.0
    re_entry_count       = 0
    selected_strike_info = None
    cooldown_until       = None

    # Load Model Arjun
    try:
        arjun     = arjun_model.load_arjun_model()
        log.info("Model Arjun loaded for live monitoring.")
    except Exception as e:
        arjun = None
        log.warning(f"Failed to load Model Arjun ({e}). Only static SL/VWAP will be used.")

    print(f"\n{'='*60}")
    print(f"  NIFTY STRADDLE VWAP STRATEGY — {today}  (Expiry: {expiry})")
    print(f"  Mode: {'📝 PAPER TRADE' if paper_trade else '🔴 LIVE TRADE'}")
    print(f"{'='*60}\n")

    # ── Phase 1: Wait for 9:15 IST (skip if already past) ───
    now_ist = _now_ist()
    if now_ist.time() < open_time_t:
        print(f"⏳ Waiting for 9:15 AM IST … (current IST: {now_ist.strftime('%H:%M:%S')})")
        _wait_until_ist(open_time_t)
    print(f"🔔 Market open — Fetching spot price … (IST: {_now_ist().strftime('%H:%M:%S')})")

    spot = dc.get_spot_price(data_broker)

    if spot <= 100:
        print(f"❌ ERROR: Spot price returned {spot} from {getattr(data_broker, 'name', 'data_broker')}.")
        log.error(f"Aborting: spot price={spot}")
        return

    strikes = dc.get_nifty_expiry_strikes(data_broker, spot, expiry)
    vix     = dc.get_vix(data_broker)

    if not strikes or len(strikes) == 0:
        msg = f"❌ ERROR: No strikes found for expiry {expiry} at spot {spot:.0f}."
        log.error(msg)
        print(msg)
        return

    atm_idx = len(strikes) // 2
    atm_val = strikes[atm_idx]['atm']

    try:
        prev_close = dc.get_nifty_prev_close(data_broker, today)
        dc.save_market_context(data_broker, today, expiry, atm_val, spot, prev_close, vix)
        log.info(f"Saved market context: spot={spot:.0f}, vix={vix:.2f}")
    except Exception as e:
        log.error(f"Failed to save market context: {e}")

    print(f"   Nifty Spot: {spot:.0f} | ATM: {atm_val} | VIX: {vix:.2f}")
    print(f"   Scanning {len(strikes)} strikes: {[s['strike'] for s in strikes]}\n")

    snapshot = dc.get_live_snapshot(data_broker, strikes)
    for _, row in snapshot.iterrows():
        vol = row["ce_volume"] + row["pe_volume"]
        vwap_tracker.update(row["strike"], row["straddle_price"], max(vol, 1))

    # ── Phase 2: RESUME STATE or SELECT STRIKE ─────────────────
    state = load_trading_state()
    should_resume = False
    
    if state:
        print(f"\n{'─'*60}")
        print(f"  💾 Saved state found for today:")
        print(f"     Strike : {state['strike']}")
        print(f"     Updated: {state['updated_at']}")
        print(f"{'─'*60}")
        try:
            resume_input = input("  Resume from saved state? (Y/n): ").strip().lower()
        except EOFError:
            resume_input = 'y'  # Auto-resume if non-interactive

        if resume_input in ('', 'y', 'yes'):
            should_resume = True
            print(f"🔄 RESUMING from Strike {state['strike']} (updating VWAP tracker history...)")
        else:
            clear_trading_state()
            print("🆕 STARTING FRESH — running ML strike selection...")

    if should_resume and state:
        best_strike = state["strike"]
        initial_pv  = state["cum_pv"]
        initial_vol = state["cum_vol"]
        # Seed the vwap_tracker with loaded history
        vwap_tracker.update(best_strike, initial_pv/initial_vol if initial_vol > 0 else 0, initial_vol)
        print(f"🔄 Resuming from Strike {best_strike} (VWAP history: {initial_vol:,.0f} vol)")
        log.info(f"Resuming Strike {best_strike} | PV Sum: {initial_pv:.2f} | Vol: {initial_vol}")
    else:
        # SELECT NEW STRIKE — Wait until scan_time (e.g., 9:20 AM)
        if now_ist.time() < scan_time_t:
            print(f"🤖 Waiting for {config.SCAN_TIME} AM IST … (current IST: {now_ist.strftime('%H:%M:%S')})")
            _wait_until_ist(scan_time_t)

        print(f"🤖 Running ML model for strike selection … (IST: {_now_ist().strftime('%H:%M:%S')})")
        
        snapshot_920      = dc.get_live_snapshot(data_broker, strikes)
        vwap_map          = vwap_tracker.get_all()
        nifty_prev_change = 0.0
        nifty_open_gap    = 0.0

        try:
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT nifty_prev_day_change, nifty_open_gap
                        FROM market_context WHERE trade_date = %s
                    """, (today,))
                    row = cur.fetchone()
                    if row:
                        nifty_prev_change = float(row[0] or 0)
                        nifty_open_gap    = float(row[1] or 0)
        except Exception:
            pass

        try:
            prediction   = ml.predict_best_strike(
                snapshot_df=snapshot_920, vix=vix,
                nifty_prev_change=nifty_prev_change, nifty_open_gap=nifty_open_gap,
                vwap_map=vwap_map, trade_date=today, expiry_date=expiry
            )
            # Handle potential None or missing key from predict_best_strike
            best_strike = (prediction.get("best_strike") if prediction else None) or strikes[atm_idx]["strike"]
        except Exception as e:
            log.warning(f"ML prediction failed ({e}), falling back to ATM.")
            best_strike  = strikes[atm_idx]["strike"]

        save_trading_state(best_strike, 0.0, 0) # Initialize state file

    selected_strike_info = next((s for s in strikes if s["strike"] == best_strike), strikes[atm_idx])
    print(f"\n📌 Selected Strike: {best_strike}\n")

    # ── Phase 3: 9:20–15:00 IST — Monitor & Trade ────────────
    print("👁️  Monitoring VWAP crossovers …\n")

    last_minute    = None
    cumulative_pnl = 0.0
    trade_count    = 0
    cooldown_until = None
    
    # Note: VWAP is handled by vwap_tracker (already seeded if resumed)

    while True:
        now = _now_ist()                               # ✅ always IST-aware
        now_t = now.time()

        # ── Hard exit at 15:00 IST ────────────────────────────
        if now_t >= dt.time(15, 15):
            print("🏁 15:15 IST | Market closing. Finalizing for the day.")
            break

        # ── Daily guardrails ──────────────────────────────────
        if position is None or not position.is_open:
            if trade_count >= config.MAX_TRADES_PER_DAY:
                print("🛑 Daily trade limit reached. Stopping for today.")
                break
            if cumulative_pnl <= config.DAILY_STOP_LOSS:
                print(f"🛑 Daily Stop Loss hit (₹{cumulative_pnl:,.0f}). Stopping for today.")
                break
            if trade_count >= config.MAX_TRADES_FOR_PROFIT_TARGET and cumulative_pnl >= config.DAILY_PROFIT_TARGET:
                print(f"✅ Daily Profit Target reached (₹{cumulative_pnl:,.0f}). Stopping for today.")
                break

        # ── Force exit open position at hard_exit_time ───────
        if position is not None and position.is_open and now_t >= hard_exit_time:
            snap = dc.get_live_snapshot(data_broker, [selected_strike_info])
            cur_price = snap.iloc[0]["straddle_price"]
            pnl = _exit_position(active_brokers, position, cur_price, "TIME_EXIT", paper_trade, now)
            cumulative_pnl += pnl
            log_trade_to_db(position, now, cur_price, "TIME_EXIT", pnl, expiry, paper_trade)
            position = None
            break

        # ── Throttle to once per minute ───────────────────────
        curr_min = now.replace(second=0, microsecond=0)
        if curr_min == last_minute:
            time.sleep(5)
            continue
        last_minute = curr_min

        # ── Fetch live quote ──────────────────────────────────
        try:
            snap     = dc.get_live_snapshot(data_broker, [selected_strike_info])
            cur_row  = snap.iloc[0]
            straddle = cur_row["straddle_price"]
            vol      = cur_row["ce_volume"] + cur_row["pe_volume"]
            vwap_tracker.update(selected_strike_info["strike"], straddle, max(vol, 1))
            vwap     = vwap_tracker.get_vwap(selected_strike_info["strike"])
            
            # Persist state (save cum_pv and cum_vol for accuracy)
            save_trading_state(
                selected_strike_info["strike"],
                vwap_tracker._cum_pv[selected_strike_info["strike"]],
                vwap_tracker._cum_vol[selected_strike_info["strike"]]
            )
        except Exception as e:
            log.error(f"Quote fetch error: {e}")
            time.sleep(10)
            continue

        gap_pct    = (straddle - vwap) / vwap * 100 if vwap else 0
        in_cooldown = cooldown_until and now < cooldown_until
        cooldown_str = f" [COOLDOWN until {cooldown_until.strftime('%H:%M')}]" if in_cooldown else ""

        # ✅ FIX: was `t.strftime` (undefined var) — use `now`
        print(f"  {now.strftime('%H:%M')} IST | Strike {selected_strike_info['strike']} | "
              f"Straddle: {straddle:.1f} | VWAP: {vwap:.1f} | Gap: {gap_pct:+.2f}%"
              + (" [POSITION OPEN]" if position and position.is_open else "")
              + cooldown_str)

        # ── Entry logic ───────────────────────────────────────
        if position is None or not position.is_open:
            if straddle < vwap and not in_cooldown:
                if paper_trade or check_margins(active_brokers, lots):
                    position = Position(
                        strike=selected_strike_info["strike"],
                        expiry=dt.datetime.strptime(selected_strike_info["expiry"], "%Y-%m-%d").date(),
                        ce_symbol=selected_strike_info["ce_symbol"],
                        pe_symbol=selected_strike_info["pe_symbol"],
                        entry_premium=straddle,
                        entry_time=now,
                        lots=lots
                    )
                    position.re_entry_count = re_entry_count
                    trade_count += 1
                    if paper_trade:
                        position.is_open = True
                        print(f"  📝 [PAPER] SELL straddle {position.strike} @ {straddle:.1f}")
                    else:
                        if place_sell_straddle(active_brokers, position):
                            position.is_open = True
                        else:
                            # FAILED to place. Reset position object so it can retry later.
                            print(f"  ⚠️  [LIVE] Entry FAILED. See logs. Retrying next minute.")
                            position = None

        # ── Exit logic ────────────────────────────────────────
        else:
            # Stop Loss
            if position.sl_hit(straddle):
                pnl = _exit_position(active_brokers, position, straddle, "STOP_LOSS", paper_trade, now)
                cumulative_pnl += pnl
                total_pnl      += pnl
                log_trade_to_db(position, now, straddle, "STOP_LOSS", pnl, expiry, paper_trade)
                position       = None
                cooldown_until = now + dt.timedelta(minutes=config.SL_COOLDOWN_MINUTES)
                print(f"  ❄️  SL HIT. Cooldown until {cooldown_until.strftime('%H:%M IST')}")

                if config.REENTRY_ALLOWED and config.REENTRY_REEVALUATE_STRIKE:
                    selected_strike_info = _reevaluate_strike(
                        data_broker, strikes, vwap_tracker, vix,
                        nifty_prev_change, nifty_open_gap, today, expiry
                    )
                re_entry_count += 1
                continue

            # Model Arjun dynamic exit
            if arjun is not None:
                pnl_pts  = position.entry_premium - straddle
                max_pnl  = max(getattr(position, 'max_pnl_pts', 0), pnl_pts)
                position.max_pnl_pts = max_pnl
                drawdown = max_pnl - pnl_pts

                curr_iv  = float(cur_row.get("ce_iv", 0)) + float(cur_row.get("pe_iv", 0))
                if not hasattr(position, 'entry_iv'):
                    position.entry_iv = curr_iv
                iv_d     = curr_iv - position.entry_iv
                delta_d  = abs(float(cur_row.get("ce_delta", 0)) + float(cur_row.get("pe_delta", 0)))
                theta_v  = float(cur_row.get("ce_theta", 0)) + float(cur_row.get("pe_theta", 0))

                if not hasattr(position, 'vol_window'):
                    position.vol_window = []
                position.vol_window.append(vol)
                if len(position.vol_window) > 15:
                    position.vol_window.pop(0)
                avg_vol = max(sum(position.vol_window) / len(position.vol_window), 1.0)
                rel_vol = min(vol / avg_vol, 100.0)

                result = arjun_model.predict_exit(
                    pnl_pts=pnl_pts, max_pnl_so_far=max_pnl, drawdown=drawdown,
                    vwap_gap_pct=gap_pct, delta_drift=delta_d,
                    theta_velocity=theta_v, iv_drift=iv_d, rel_vol_15m=rel_vol,
                    model=arjun, threshold=config.ARJUN_EXIT_THRESHOLD
                )

                if result["should_exit"]:
                    reason = f"ARJUN_EXIT ({result['confidence']:.0%})"
                    pnl    = _exit_position(active_brokers, position, straddle, reason, paper_trade, now)
                    cumulative_pnl += pnl
                    total_pnl      += pnl
                    log_trade_to_db(position, now, straddle, reason, pnl, expiry, paper_trade)
                    position = None

                    if config.REENTRY_ALLOWED and config.REENTRY_REEVALUATE_STRIKE:
                        selected_strike_info = _reevaluate_strike(
                            data_broker, strikes, vwap_tracker, vix,
                            nifty_prev_change, nifty_open_gap, today, expiry
                        )
                    re_entry_count += 1
                    continue

            # VWAP crossover exit
            if straddle > vwap:
                pnl = _exit_position(active_brokers, position, straddle, "VWAP_CROSS", paper_trade, now)
                cumulative_pnl += pnl
                total_pnl      += pnl
                log_trade_to_db(position, now, straddle, "VWAP_CROSS", pnl, expiry, paper_trade)
                position = None

                if config.REENTRY_ALLOWED and config.REENTRY_REEVALUATE_STRIKE:
                    selected_strike_info = _reevaluate_strike(
                        data_broker, strikes, vwap_tracker, vix,
                        nifty_prev_change, nifty_open_gap, today, expiry
                    )
                re_entry_count += 1

        time.sleep(10)

    print(f"\n{'='*60}")
    print(f"  📊 END OF DAY SUMMARY — {today}")
    print(f"  Total P&L  : ₹{total_pnl:,.0f}")
    print(f"  Re-entries : {re_entry_count}")
    print(f"{'='*60}\n")
    log.info(f"EOD: total_pnl={total_pnl:.0f}, re_entries={re_entry_count}")


# ── Helpers ──────────────────────────────────────────────────

def _exit_position(brokers, position: Position, current_premium: float,
                   reason: str, paper_trade: bool, now: dt.datetime) -> float:
    pnl = position.pnl_points(current_premium)
    position.is_open = False
    if paper_trade:
        print(f"  📝 [PAPER] BUY straddle {position.strike} @ {current_premium:.1f} | "
              f"Reason: {reason} | P&L: ₹{pnl:,.0f}")
    else:
        place_buy_straddle(brokers, position, reason, current_premium)
    return pnl


def _reevaluate_strike(data_broker, strikes, vwap_tracker, vix,
                       prev_change, open_gap, today, expiry):
    snap     = dc.get_live_snapshot(data_broker, strikes)
    vwap_map = vwap_tracker.get_all()
    try:
        pred = ml.predict_best_strike(
            snapshot_df=snap, vix=vix,
            nifty_prev_change=prev_change, nifty_open_gap=open_gap,
            vwap_map=vwap_map, trade_date=today, expiry_date=expiry
        )
        best = pred["best_strike"]
        print(f"  🔄 Re-entry strike re-evaluated → {best}")
    except Exception:
        best = strikes[len(strikes) // 2]["strike"]
    return next((s for s in strikes if s["strike"] == best), strikes[len(strikes) // 2])


def _wait_until_ist(target_time: dt.time):
    """
    Block until IST wall clock reaches target_time.
    ✅ FIX: was using naive dt.datetime.now() which gives UTC on many servers.
    Now always compares in IST.
    """
    while True:
        now_ist = _now_ist()
        if now_ist.time() >= target_time:
            return
        target_dt = dt.datetime.combine(now_ist.date(), target_time, tzinfo=IST)
        remaining = (target_dt - now_ist).total_seconds()
        if remaining <= 0:
            return
        wait = 30 if remaining > 60 else 5
        print(f"   ⏳ {now_ist.strftime('%H:%M:%S')} IST — waiting {remaining/60:.1f} min for {target_time} …")
        time.sleep(wait)
