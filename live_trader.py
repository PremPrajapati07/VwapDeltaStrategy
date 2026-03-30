# ============================================================
#  live_trader.py — Live Trading Engine
# ============================================================
"""
Responsibilities:
  1. At 9:15 → Connect Kite, start VWAP tracker for all 11 strikes
  2. At 9:20 → Run ML model → select best strike
  3. Every minute → check VWAP crossover → entry / exit / re-entry
  4. Re-entry → re-evaluate strike via ML
  5. At 3:00 PM → force close all positions
  6. Log every trade to PostgreSQL trade_log table
"""

import os
import time
import logging
import datetime as dt
import pandas as pd
from collections import defaultdict

import config
import data_collector as dc
import ml_model as ml
import db

os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, "live_trader.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


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
                 entry_premium: float, entry_time: dt.datetime,
                 lots: int = 1, lot_size: int = None):
        self.strike         = strike
        self.ce_symbol      = ce_symbol
        self.pe_symbol      = pe_symbol
        self.entry_premium  = entry_premium
        self.entry_time     = entry_time
        self.lots           = lots
        self.lot_size       = lot_size or config.LOT_SIZE
        self.sl_level       = entry_premium * config.SL_MULTIPLIER
        self.re_entry_count = 0
        self.is_open        = True

    def pnl_points(self, current_premium: float) -> float:
        return (self.entry_premium - current_premium) * self.lots * self.lot_size

    def sl_hit(self, current_premium: float) -> bool:
        return current_premium >= self.sl_level


# ── Order Execution ──────────────────────────────────────────

def place_sell_straddle(kite, position: Position):
    """Place sell orders for CE and PE legs."""
    try:
        ce_order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=position.ce_symbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=position.lots * 50,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET,
        )
        pe_order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=position.pe_symbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=position.lots * config.LOT_SIZE,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET,
        )
        log.info(f"SELL straddle {position.strike}: CE={ce_order}, PE={pe_order}")
        print(f"  ✅ SOLD   Strike {position.strike} | Premium {position.entry_premium:.1f}")
        return True
    except Exception as e:
        log.error(f"Order placement failed: {e}")
        print(f"  ❌ Order FAILED: {e}")
        return False


def place_buy_straddle(kite, position: Position, exit_reason: str, current_premium: float):
    """Place buy (close) orders for CE and PE legs."""
    try:
        kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=position.ce_symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=position.lots * 50,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET,
        )
        kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=position.pe_symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=position.lots * config.LOT_SIZE,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET,
        )
        pnl = position.pnl_points(current_premium)
        log.info(f"BUY straddle {position.strike} | reason={exit_reason} | P&L={pnl:.0f}")
        print(f"  🔁 CLOSED Strike {position.strike} | Reason: {exit_reason} | P&L: ₹{pnl:,.0f}")
        return pnl
    except Exception as e:
        log.error(f"Close order failed: {e}")
        print(f"  ❌ Close FAILED: {e}")
        return 0.0


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

def check_margin(kite, lot_size_nifty: int = 50, lots: int = 1) -> bool:
    """Check if available cash is enough for the required margin."""
    if not config.CHECK_MARGIN:
        return True
    
    try:
        margins = kite.margins()
        available_cash = margins["equity"]["available"]["cash"]
        required = config.MIN_REQUIRED_MARGIN * lots
        
        if available_cash < required:
            log.warning(f"Insufficient Margin: Available ₹{available_cash:,.0f}, Required ₹{required:,.0f}")
            print(f"  ⚠️  INSUFFICIENT MARGIN | Available: ₹{available_cash:,.0f} | Required: ₹{required:,.0f}")
            return False
        return True
    except Exception as e:
        log.error(f"Margin check failed: {e}")
        return True # Default to True on error to not block trading if API fails


# ── Main Trading Loop ────────────────────────────────────────

def run_live_trading(kite, lots: int = 1, paper_trade: bool = True):
    """
    Main entry point for live trading day.
    kite: authenticated KiteConnect instance (from kite_auth.get_kite())
    """
    today     = dt.date.today()
    expiry    = dc.get_nearest_expiry(kite)
    hard_exit = dt.time(*[int(x) for x in config.HARD_EXIT_TIME.split(":")])
    scan_time = dt.time(*[int(x) for x in config.SCAN_TIME.split(":")])

    vwap_tracker         = VWAPTracker()
    position             = None
    total_pnl            = 0.0
    re_entry_count       = 0
    selected_strike_info = None
    cooldown_until       = None

    print(f"\n{'='*60}")
    print(f"  NIFTY STRADDLE VWAP STRATEGY — {today}  (Expiry: {expiry})")
    print(f"  Mode: {'📝 PAPER TRADE' if paper_trade else '🔴 LIVE TRADE'}")
    print(f"{'='*60}\n")

    # ── Phase 1: 9:15 — initialize strikes ──
    print("⏳ 9:15 — Fetching spot price and building strike list …")
    _wait_until(dt.time(9, 15))
    spot    = dc.get_spot_price(kite)
    strikes = dc.get_nifty_expiry_strikes(kite, spot, expiry)
    vix     = dc.get_vix(kite)
    
    # Save today's market context (needed for ML features at 9:20)
    try:
        prev_close = dc.get_nifty_prev_close(kite, today)
        dc.save_market_context(kite, today, expiry, strikes[5]['atm'], spot, prev_close, vix)
        log.info(f"Saved market context for {today}: spot={spot:.0f}, vix={vix:.2f}")
    except Exception as e:
        log.error(f"Failed to save market context: {e}")

    print(f"   Nifty Spot: {spot:.0f} | ATM: {strikes[5]['atm']} | VIX: {vix:.2f}")
    print(f"   Scanning {len(strikes)} strikes: {[s['strike'] for s in strikes]}\n")

    snapshot = dc.get_live_snapshot(kite, strikes)
    for _, row in snapshot.iterrows():
        vol = row["ce_volume"] + row["pe_volume"]
        vwap_tracker.update(row["strike"], row["straddle_price"], max(vol, 1))

    # ── Phase 2: 9:20 — ML strike selection ──
    print("🤖 9:20 — Running ML model for strike selection …")
    _wait_until(scan_time)

    snapshot_920      = dc.get_live_snapshot(kite, strikes)
    vwap_map          = vwap_tracker.get_all()
    nifty_prev_change = 0.0
    nifty_open_gap    = 0.0

    # Try to load prev change and open gap from market_context
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT nifty_prev_day_change, nifty_open_gap
                    FROM market_context
                    WHERE trade_date = %s
                """, (today,))
                row = cur.fetchone()
                if row:
                    nifty_prev_change = float(row[0] or 0)
                    nifty_open_gap    = float(row[1] or 0)
    except Exception:
        pass

    try:
        prediction = ml.predict_best_strike(
            snapshot_df=snapshot_920,
            vix=vix,
            nifty_prev_change=nifty_prev_change,
            nifty_open_gap=nifty_open_gap,
            vwap_map=vwap_map,
            trade_date=today,
            expiry_date=expiry
        )
        best_strike = prediction["best_strike"]
    except Exception as e:
        log.warning(f"ML prediction failed ({e}), falling back to ATM.")
        best_strike = strikes[5]["strike"]

    selected_strike_info = next((s for s in strikes if s["strike"] == best_strike), strikes[5])
    print(f"\n📌 Selected Strike: {best_strike}\n")

    # ── Phase 3: 9:20–15:00 — Monitor & Trade ──
    print("👁️  Monitoring VWAP crossovers …\n")
    last_minute = None

    while True:
        now = dt.datetime.now()
        t   = now.time()

        if t >= hard_exit:
            if position is not None and position.is_open:
                snap = dc.get_live_snapshot(kite, [selected_strike_info])
                cur  = snap.iloc[0]["straddle_price"]
                pnl  = _exit_position(kite, position, cur, "TIME_EXIT", paper_trade, now)
                total_pnl += pnl
                log_trade_to_db(position, now, cur, "TIME_EXIT", pnl, expiry, paper_trade)
                position = None
            break

        curr_min = now.replace(second=0, microsecond=0)
        if curr_min == last_minute:
            time.sleep(5)
            continue
        last_minute = curr_min

        try:
            snap     = dc.get_live_snapshot(kite, [selected_strike_info])
            cur_row  = snap.iloc[0]
            straddle = cur_row["straddle_price"]
            vol      = cur_row["ce_volume"] + cur_row["pe_volume"]
            vwap_tracker.update(selected_strike_info["strike"], straddle, max(vol, 1))
            vwap     = vwap_tracker.get_vwap(selected_strike_info["strike"])
        except Exception as e:
            log.error(f"Quote fetch error: {e}")
            time.sleep(10)
            continue

        gap_pct = (straddle - vwap) / vwap * 100 if vwap else 0
        
        # Check cooldown
        in_cooldown = cooldown_until and now < cooldown_until
        cooldown_str = f" [COOLDOWN until {cooldown_until.strftime('%H:%M')}]" if in_cooldown else ""

        print(f"  {t.strftime('%H:%M')} | Strike {selected_strike_info['strike']} | "
              f"Straddle: {straddle:.1f} | VWAP: {vwap:.1f} | Gap: {gap_pct:+.2f}%"
              + (" [POSITION OPEN]" if position and position.is_open else "")
              + cooldown_str)

        if position is None or not position.is_open:
            if straddle < vwap and not in_cooldown:
                if paper_trade or check_margin(kite, 50, lots):
                    entry_premium = straddle
                    position = Position(
                        strike=selected_strike_info["strike"],
                        ce_symbol=selected_strike_info["ce_symbol"],
                        pe_symbol=selected_strike_info["pe_symbol"],
                        entry_premium=entry_premium,
                        entry_time=now,
                        lots=lots
                    )
                    position.re_entry_count = re_entry_count
                    if paper_trade:
                        print(f"  📝 [PAPER] SELL straddle {position.strike} @ {entry_premium:.1f}")
                    else:
                        place_sell_straddle(kite, position)
        else:
            if position.sl_hit(straddle):
                pnl = _exit_position(kite, position, straddle, "STOP_LOSS", paper_trade, now)
                total_pnl += pnl
                log_trade_to_db(position, now, straddle, "STOP_LOSS", pnl, expiry, paper_trade)
                position = None
                
                # Start cooldown
                cooldown_until = now + dt.timedelta(minutes=config.SL_COOLDOWN_MINUTES)
                print(f"  ❄️  SL HIT. Cooldown until {cooldown_until.strftime('%H:%M:%S')}")

                if config.REENTRY_ALLOWED and config.REENTRY_REEVALUATE_STRIKE:
                    selected_strike_info = _reevaluate_strike(
                        kite, strikes, vwap_tracker, vix,
                        nifty_prev_change, nifty_open_gap, today, expiry
                    )
                re_entry_count += 1
                continue

            if straddle > vwap:
                pnl = _exit_position(kite, position, straddle, "VWAP_CROSS", paper_trade, now)
                total_pnl += pnl
                log_trade_to_db(position, now, straddle, "VWAP_CROSS", pnl, expiry, paper_trade)
                position = None

                if config.REENTRY_ALLOWED and config.REENTRY_REEVALUATE_STRIKE:
                    selected_strike_info = _reevaluate_strike(
                        kite, strikes, vwap_tracker, vix,
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

def _exit_position(kite, position: Position, current_premium: float,
                   reason: str, paper_trade: bool, now: dt.datetime) -> float:
    pnl = position.pnl_points(current_premium)
    position.is_open = False
    if paper_trade:
        print(f"  📝 [PAPER] BUY straddle {position.strike} @ {current_premium:.1f} | "
              f"Reason: {reason} | P&L: ₹{pnl:,.0f}")
    else:
        place_buy_straddle(kite, position, reason, current_premium)
    return pnl


def _reevaluate_strike(kite, strikes, vwap_tracker, vix,
                       prev_change, open_gap, today, expiry):
    snap     = dc.get_live_snapshot(kite, strikes)
    vwap_map = vwap_tracker.get_all()
    try:
        pred = ml.predict_best_strike(
            snapshot_df=snap,
            vix=vix,
            nifty_prev_change=prev_change,
            nifty_open_gap=open_gap,
            vwap_map=vwap_map,
            trade_date=today,
            expiry_date=expiry
        )
        best = pred["best_strike"]
        print(f"  🔄 Re-entry strike re-evaluated → {best}")
    except Exception:
        best = strikes[5]["strike"]
    return next((s for s in strikes if s["strike"] == best), strikes[5])


def _wait_until(target_time: dt.time):
    """Block until the wall clock reaches target_time."""
    while True:
        now = dt.datetime.now().time()
        if now >= target_time:
            return
        remaining = (
            dt.datetime.combine(dt.date.today(), target_time) -
            dt.datetime.combine(dt.date.today(), now)
        ).seconds
        time.sleep(30 if remaining > 60 else 5)
