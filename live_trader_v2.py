# ============================================================
#  live_trader_v2.py — Live Trading Engine (V2 Variant)
# ============================================================
"""
Responsibilities:
  1. At 9:15 IST → Connect Kite, start VWAP tracker for all 11 strikes
  2. At 9:20 IST → Run Krishna V2 ML model → select best strike
  3. Every minute → check VWAP crossover → entry / exit / re-entry
  4. Re-entry → re-evaluate strike via ML (V2)
  5. At 3:15 PM IST → force close all positions
  6. Log every trade to PostgreSQL trade_log table
"""

import os
import json
import time
import logging
import datetime as dt
import pandas as pd
from collections import defaultdict
from zoneinfo import ZoneInfo

import config
import data_collector as dc
import krishna_v2_model as ml_v2
import arjun_model_v3 as arjun_v3
import db
import kite_auth
import kotak_auth
from brokers import KiteBroker, NeoBroker

IST = ZoneInfo("Asia/Kolkata")

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


# ── Session Tracker ──────────────────────────────────────────

class SessionTracker:
    """
    Tracks per-strike performance for blacklisting.
    V2/V3 models are not adaptive, so this is used only for cooldowns.
    """
    BLACKLIST_LOSS_PTS = -7.0
    BLACKLIST_MINS     = 60

    def __init__(self):
        self.session_pnl          = 0.0
        self.trades_taken         = 0
        self.strike_last_pnl:     dict = {}
        self.strike_consec_losses:dict = {}
        self._blacklist:          dict = {}

    def record_trade(self, strike: int, pnl_pts: float, now: dt.datetime):
        self.session_pnl                    += pnl_pts
        self.trades_taken                   += 1
        self.strike_last_pnl[strike]         = pnl_pts
        if pnl_pts < 0:
            self.strike_consec_losses[strike] = self.strike_consec_losses.get(strike, 0) + 1
            if pnl_pts <= self.BLACKLIST_LOSS_PTS:
                until = now + dt.timedelta(minutes=self.BLACKLIST_MINS)
                self._blacklist[strike] = until
                log.info(f"V2: Strike {strike} BLACKLISTED until {until.strftime('%H:%M')}")
        else:
            self.strike_consec_losses[strike] = 0

    def is_blacklisted(self, strike: int, now: dt.datetime) -> bool:
        until = self._blacklist.get(strike)
        if until and now < until: return True
        return False


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
        self.is_open        = False
        self.is_safe_mode   = False
        self.max_pnl_pts    = 0.0
        self.vol_window     = []
        self.entry_iv       = 0.0

    def sl_hit(self, current_premium: float) -> bool:
        return current_premium >= self.sl_level


# ── Order Execution ──────────────────────────────────────────

def place_sell_straddle(brokers, position: Position):
    if not brokers: return False
    all_success = True
    for broker in brokers:
        try:
            log.info(f"[{broker.name}] V2: Placing SELL straddle for {position.strike}")
            ce_id = broker.place_order(position.ce_symbol, position.lots * position.lot_size, "SELL", strike=position.strike, expiry=position.expiry)
            pe_id = broker.place_order(position.pe_symbol, position.lots * position.lot_size, "SELL", strike=position.strike, expiry=position.expiry)
            if ce_id and pe_id:
                print(f"  ✅ [{broker.name}] SOLD Strike {position.strike} (V2)")
            else:
                all_success = False
        except Exception as e:
            all_success = False
            log.error(f"[{broker.name}] Order placement exception: {e}")
    return all_success


def _exit_position(brokers, position: Position, current_premium: float, reason: str, paper_trade: bool, now: dt.datetime) -> float:
    pnl_pts = position.entry_premium - current_premium
    pnl_inr = pnl_pts * position.lots * position.lot_size
    position.is_open = False
    if paper_trade:
        print(f"  📝 [PAPER V2] BUY close {position.strike} @ {current_premium:.1f} | Reason: {reason} | P&L: {pnl_pts:+.1f} pts")
    else:
        for broker in brokers:
            broker.place_order(position.ce_symbol, position.lots * position.lot_size, "BUY", strike=position.strike, expiry=position.expiry)
            broker.place_order(position.pe_symbol, position.lots * position.lot_size, "BUY", strike=position.strike, expiry=position.expiry)
            print(f"  🔁 [{broker.name}] CLOSED Strike {position.strike} (V2) | P&L: {pnl_pts:+.1f} pts")
    return pnl_pts


# ── Main Loop ────────────────────────────────────────────────

def run_live_trading(lots: int = 1, paper_trade: bool = True, skip_kite: bool = False, skip_neo: bool = False):
    today = _now_ist().date()
    data_broker = None
    active_brokers = []

    # Initialize Zerodha for Data
    try:
        kite_inst = kite_auth.get_kite()
        data_broker = KiteBroker(kite_inst)
        log.info("Zerodha connected for V2 data.")
    except Exception as e:
        print(f"❌ Zerodha connection failed: {e}")
        return

    # Initialize Kotak for Execution
    if not paper_trade:
        try:
            neo_inst = kotak_auth.get_neo_client()
            if neo_inst:
                active_brokers = [NeoBroker(neo_inst)]
                log.info("Kotak Neo connected for V2 execution.")
        except Exception as e:
            print(f"❌ Kotak Neo connection failed: {e}")
            return

    expiry = dc.get_nearest_expiry(data_broker)
    scan_time_t = dt.time(9, 20)
    hard_exit_time = dt.time(15, 0)

    vwap_tracker = VWAPTracker()
    session_tracker = SessionTracker()
    position = None
    cumulative_pnl_pts = 0.0
    trade_count = 0
    selected_strike_info = None
    cooldown_until = None

    # Load V2/V3 Models
    try:
        ml_v2_model = ml_v2.load_v2_model()
        arjun_v3_model = arjun_v3.load_arjun_model()
        log.info("Krishna V2 and Arjun V3 models loaded.")
    except Exception as e:
        print(f"❌ Failed to load V2/V3 models: {e}")
        return

    print(f"\n{'='*60}")
    print(f"  NIFTY STRADDLE V2 ENGINE — {today}  (Expiry: {expiry})")
    print(f"  Mode: {'📝 PAPER' if paper_trade else '🔴 REAL (Kotak)'}")
    print(f"{'='*60}\n")

    # Wait for 9:15
    now_ist = _now_ist()
    if now_ist.time() < dt.time(9, 15):
        dc._wait_until_ist(dt.time(9, 15))

    spot = dc.get_spot_price(data_broker)
    strikes = dc.get_nifty_expiry_strikes(data_broker, spot, expiry)
    vix = dc.get_vix(data_broker)
    prev_close = dc.get_nifty_prev_close(data_broker, today)

    # Market context for ML
    snapshot_915 = dc.get_live_snapshot(data_broker, strikes)
    spot_915 = float(snapshot_915.iloc[0]["synthetic_spot"])
    
    # Logic Parity Fix: Both V2 and V3 models use (Open/9:15 Spot - Prev Close) in training context
    nifty_prev_change = (spot_915 - prev_close) / prev_close * 100 if prev_close else 0
    
    # Logic Parity Fix: Use actual Nifty Open instead of proxy
    nifty_open = dc.get_nifty_open(data_broker, today)
    nifty_open_gap = (nifty_open - prev_close) / prev_close * 100 if prev_close and nifty_open else nifty_prev_change

    # Anchor 9:20 baseline once
    baseline_920 = {}

    # Initial Strike Selection at 9:20
    if now_ist.time() < scan_time_t:
        dc._wait_until_ist(scan_time_t)

    print("🤖 Running Krishna V2 Strike Selection...")
    snap_920 = dc.get_live_snapshot(data_broker, strikes)
    vwap_map = {r["strike"]: r["straddle_price"] for _, r in snap_920.iterrows()} # Seed VWAP
    
    # Store baseline for V2 model
    for _, r in snap_920.iterrows():
        baseline_920[r["strike"]] = {"spot": r["synthetic_spot"], "straddle": r["straddle_price"]}

    pred = ml_v2.predict_best_strike_v2(
        snapshot_df=snap_920, vwap_map=vwap_map, baseline_920=baseline_920,
        current_time=scan_time_t, vix=vix, nifty_prev_change=nifty_prev_change,
        nifty_open_gap=nifty_open_gap, trade_date=today, expiry_date=expiry,
        model=ml_v2_model
    )
    best_strike = pred["best_strike"]
    selected_strike_info = next(s for s in strikes if s["strike"] == best_strike)
    print(f"🎯 Selected V2 Strike: {best_strike} (Conf: {pred['confidence']:.1%})")

    # Monitoring Loop
    while True:
        now = _now_ist()
        if now.time() >= dt.time(15, 15): break

        # Throttle
        if now.second % 60 != 0:
            time.sleep(1)
            continue

        try:
            snap = dc.get_live_snapshot(data_broker, [selected_strike_info])
            cur_row = snap.iloc[0]
            straddle = cur_row["straddle_price"]
            vol = cur_row["ce_volume"] + cur_row["pe_volume"]
            vwap_tracker.update(best_strike, straddle, max(vol, 1))
            vwap = vwap_tracker.get_vwap(best_strike)
        except Exception:
            time.sleep(5)
            continue

        gap_pct = (straddle - vwap) / vwap * 100 if vwap else 0
        in_cooldown = cooldown_until and now < cooldown_until
        
        print(f"  {now.strftime('%H:%M')} | {best_strike} | Str: {straddle:.1f} | VWAP: {vwap:.1f} | Gap: {gap_pct:+.2f}%")

        # Entry
        if not position:
            if trade_count < config.MAX_TRADES_PER_DAY and not in_cooldown:
                if straddle <= vwap * (1.0 - config.VWAP_ENTRY_THRESHOLD_PCT / 100.0):
                    position = Position(best_strike, selected_strike_info["ce_symbol"], selected_strike_info["pe_symbol"], straddle, now, expiry, lots)
                    position.entry_iv = float(cur_row.get("ce_iv", 0)) + float(cur_row.get("pe_iv", 0))
                    if paper_trade or place_sell_straddle(active_brokers, position):
                        position.is_open = True
                        trade_count += 1

        # Exit
        else:
            pnl_pts = position.entry_premium - straddle
            position.max_pnl_pts = max(position.max_pnl_pts, pnl_pts)
            dd = position.max_pnl_pts - pnl_pts
            
            # Safe Mode logic (V3)
            if not position.is_safe_mode and pnl_pts >= 30.0:
                position.is_safe_mode = True
                print("  🛡️ V2 Safe Mode Active")

            exit_reason = None
            if position.sl_hit(straddle): exit_reason = "STOP_LOSS"
            elif position.is_safe_mode and pnl_pts <= 2.0: exit_reason = "BE_SL"
            elif now.time() >= hard_exit_time: exit_reason = "TIME_EXIT"
            elif straddle > vwap: exit_reason = "VWAP_CROSS"
            else:
                # Arjun V3 AI Exit
                theta_v = float(cur_row.get("ce_theta", 0)) + float(cur_row.get("pe_theta", 0))
                iv_d = (float(cur_row.get("ce_iv", 0)) + float(cur_row.get("pe_iv", 0))) - position.entry_iv
                delta_d = abs(float(cur_row.get("ce_delta", 0)) + float(cur_row.get("pe_delta", 0)))
                
                position.vol_window.append(vol)
                if len(position.vol_window) > 15: position.vol_window.pop(0)
                avg_vol = max(sum(position.vol_window)/len(position.vol_window), 1)
                rel_vol = min(vol/avg_vol, 100.0)

                res = arjun_v3.predict_exit(
                    pnl_pts, position.max_pnl_pts, dd, gap_pct, delta_d, 
                    theta_v, iv_d, rel_vol, model=arjun_v3_model, threshold=0.55
                )
                if res["should_exit"]: exit_reason = f"ARJUN_V3 ({res['confidence']:.0%})"

            if exit_reason:
                pnl = _exit_position(active_brokers, position, straddle, exit_reason, paper_trade, now)
                cumulative_pnl_pts += pnl
                session_tracker.record_trade(best_strike, pnl, now)
                # Re-evaluate
                selected_strike_info = _reevaluate_strike_v2(data_broker, strikes, vwap_tracker, baseline_920, vix, nifty_prev_change, nifty_open_gap, today, expiry, ml_v2_model)
                best_strike = selected_strike_info["strike"]
                position = None
                cooldown_until = now + dt.timedelta(minutes=config.SL_COOLDOWN_MINUTES)

        time.sleep(1)


def _reevaluate_strike_v2(data_broker, strikes, vwap_tracker, baseline_920, vix, prev_change, open_gap, today, expiry, model):
    print("🔄 V2 Re-evaluating Strike...")
    snap = dc.get_live_snapshot(data_broker, strikes)
    vwap_map = vwap_tracker.get_all()
    pred = ml_v2.predict_best_strike_v2(snap, vwap_map, baseline_920, _now_ist().time(), vix, prev_change, open_gap, today, expiry, model=model)
    print(f"🎯 New V2 Focus: {pred['best_strike']}")
    return next(s for s in strikes if s["strike"] == pred["best_strike"])
