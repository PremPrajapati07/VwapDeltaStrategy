# 📊 Nifty VWAP Straddle Strategy — Project Summary

> **Project Type:** Algorithmic Trading System (Options Selling)
> **Instrument:** Nifty 50 Weekly Options (Thursday Expiry)
> **Exchange:** Zerodha Kite Connect API (NFO + NSE)

---

## 🗺️ What This Project Does

This is a **fully automated Nifty options straddle trading system** that:
1. Uses an **ML model (XGBoost)** to select the best strike price every morning at 9:20 AM
2. Sells a **straddle** (CE + PE) when the combined premium drops **below its VWAP**
3. Buys it back when the premium rises **above VWAP** (or stop-loss / time-exit)
4. Supports **paper trading** (simulation) and **live trading** (real Kite orders)
5. Also includes a standalone **raw data collector** for detailed Greeks/IV analysis

---

## 🗂️ File-by-File Breakdown

### `config.py` — Central Configuration
- Holds all Kite API credentials (`KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`)
- Sets strategy parameters: scan time (9:20), hard exit (3:00 PM), SL multiplier (2×)
- Defines all 13 ML feature names used across the codebase
- Specifies file paths for `data/`, `models/`, `logs/`

> ⚠️ **You still need to fill your actual API credentials here before going live.**

---

### `data_collector.py` — Historical & Live Data Engine
**Key responsibilities:**
- `get_kite_session()` — authenticates and generates an access token
- `get_nifty_expiry_strikes()` — builds the list of 11 strikes (ATM ± 5 × 50 pts) for an expiry
- `fetch_straddle_minute_data()` — fetches 1-min OHLCV for CE + PE of a single strike
- `compute_straddle_vwap()` — computes cumulative VWAP from 9:15 AM
- `backfill_history()` — full loop for all Thursdays in a date range, saves to SQLite
- `get_live_snapshot()` — fetches real-time quotes for all 22 instruments at once

**Database tables created:**
| Table | Purpose |
|---|---|
| `straddle_candles` | 1-min OHLCV + VWAP per strike per day |
| `trade_log` | Every entry/exit trade with P&L |
| `daily_features` | Per-day ML feature rows for training |

---

### `ml_model.py` — XGBoost Strike Selector
**Key responsibilities:**
- `build_features_from_db()` — reads DB candles, extracts 9:20 AM features, simulates P&L for every strike on every historical Thursday, labels the best strike per day
- `simulate_pnl()` — simulates the VWAP crossover strategy on 1-min data for a single strike; handles entries, exits, stop-loss (2× premium), and forced 3 PM close
- `train_model()` — trains XGBoost with `TimeSeriesSplit` (5-fold, no look-ahead bias), prints feature importances, saves model + label encoder to `models/`
- `predict_best_strike()` — given a live 9:20 snapshot of all 11 strikes, runs the model and returns the best strike with confidence scores

**13 Features Used:**

| Feature | What It Captures |
|---|---|
| `vwap_gap_pct` | How far straddle is from VWAP at 9:20 |
| `straddle_premium` | Total CE+PE premium (theta potential) |
| `ce_pe_ratio` | Directional bias / skew |
| `distance_from_atm` | Strike distance in points |
| `distance_from_atm_pct` | Relative moneyness % |
| `oi_ce / oi_pe` | Open Interest per leg |
| `oi_imbalance` | OI skew between CE and PE |
| `days_to_expiry` | Calendar days to expiry |
| `day_of_week` | Day of trading week |
| `vix` | India VIX (volatility regime) |
| `nifty_prev_day_change` | Previous day Nifty % change |
| `nifty_open_gap` | Today's open gap vs prev close |

> ⚠️ **Note:** `vix`, `nifty_prev_day_change`, and `nifty_open_gap` are listed as ML features in `config.py` but are **not populated in `build_features_from_db()`** — these 3 features remain as `0` unless you add code to fetch and store them during backfill.

---

### `live_trader.py` — Real-Time Trading Engine
**Key responsibilities:**
- `VWAPTracker` class — in-memory, per-strike cumulative VWAP tracker updated every quote
- `Position` class — tracks entry premium, stop-loss level, re-entry count, P&L
- `place_sell_straddle()` / `place_buy_straddle()` — wraps Kite `place_order()` calls (both legs)
- `run_live_trading()` — the main trading loop:
  - 9:15 AM: fetch spot, build strike list, initialize VWAP
  - 9:20 AM: run ML model → select best strike
  - 9:20–15:00: every minute, check VWAP crossover → entry/exit/stop-loss
  - On each re-entry: re-run ML to potentially select a different strike
  - 3:00 PM: force-close any open position, print EOD summary
- `log_trade()` — saves every completed trade to the `trade_log` SQLite table
- `_wait_until()` — blocks until a specific wall-clock time

---

### `main.py` — CLI Entry Point
The single file you run from the command line. Exposes 5 modes:

```
python main.py --mode login       # Browser login, saves .access_token
python main.py --mode backfill    # Fetch historical data to SQLite
python main.py --mode train       # Train XGBoost model
python main.py --mode backtest    # Simulate strategy, print stats
python main.py --mode live        # Live paper trade (default)
python main.py --mode live --real # Live real-money trade (with confirmation)
```

**Backtest report prints:**
- Win/Loss day count and %
- Total & average P&L points
- Best possible P&L (oracle) and capture rate
- Max drawdown, Sharpe ratio
- Strike selection accuracy (ML vs. oracle best)

---

### `raw_data_collector.py` — Standalone Greeks/IV Collector
A **separate, independent script** (not used by `main.py`). It:
- Fetches full-day 1-min OHLCV for all 22 Nifty option instruments
- Computes **Implied Volatility** via Newton-Raphson Black-Scholes solver
- Computes **Delta, Gamma, Theta, Vega** using Black-Scholes
- Writes output to a richly formatted **Excel file** (`raw_data/NIFTY_DDMMMYYYY.xlsx`)
  - 22 tabs (one per strike/type), color-coded CE=blue / PE=red
  - Row groups: OHLC | Volume | OI | Greeks + IV
- Supports `--demo` mode (no Kite needed, generates synthetic data)
- Supports `--date`, `--start/--end`, `--all-days` flags

> This is useful for research/analysis but is **not connected to the live trading pipeline**.

---

## 🔄 Recommended Workflow

```
Step 1:  Fill config.py with your Kite API key & secret
Step 2:  python main.py --mode login          → get access token
Step 3:  python main.py --mode backfill       → populate DB (2–4 hrs)
Step 4:  python main.py --mode train          → train XGBoost model
Step 5:  python main.py --mode backtest       → validate on historical data
Step 6:  python main.py --mode live           → paper trade (no real orders)
Step 7:  python main.py --mode live --real    → LIVE trading (real orders!)
```

---

## ✅ What Is Implemented

- [x] Full Kite Connect authentication flow
- [x] Historical data backfill for all Thursdays
- [x] SQLite database with 3 tables (candles, trades, features)
- [x] VWAP computation on straddle price
- [x] 13-feature ML pipeline using XGBoost classifier
- [x] TimeSeriesSplit cross-validation (no future leakage)
- [x] Model save/load with pickle
- [x] Live VWAP tracker (in-memory, per-strike)
- [x] Paper trade mode (full simulation, no orders)
- [x] Real order placement (market orders, MIS product)
- [x] Stop-loss logic (2× entry premium)
- [x] Re-entry with ML re-evaluation of strike
- [x] Hard exit at 3:00 PM
- [x] Trade logging to DB
- [x] Full backtest with Sharpe ratio, drawdown, win rate
- [x] Standalone Excel raw data collector with Greeks/IV

---

## ❌ What Is Missing / Not Yet Implemented

| Gap | Details |
|---|---|
| **VIX data storage** | `vix` feature in ML config but never fetched/saved during backfill. The `get_vix()` function exists in `data_collector.py` but is not called in `backfill_history()`. |
| **Nifty prev day change** | `nifty_prev_day_change` feature defined but never computed or stored in `build_features_from_db()`. |
| **Nifty open gap** | Same as above — `nifty_open_gap` is always 0 in training. |
| **OI data in backfill** | `oi_ce` / `oi_pe` features exist but historical OI is not stored in `straddle_candles` table (Kite's historical endpoint does not always return OI). |
| **`daily_features` table write** | The DB table is defined in `init_db()` but nothing writes to it — `build_features_from_db()` returns a DataFrame but never saves it back to DB. |
| **Login token persistence** | Access token is saved to `.access_token` file but `config.py` never reads it automatically; user must pass it explicitly or it uses `load_access_token()` manually. |
| **`files/` directory** | There's a `files/` subdirectory in the project but it's empty/unused — its purpose is unclear. |
| **No unit tests** | No `tests/` folder or test suite of any kind. |
| **No scheduler / cron** | The system must be manually launched each morning; no auto-start at 9:15 AM. |
| **No dashboard/UI** | All output is CLI-only. No Streamlit, web dashboard, or notification system (Telegram/email). |
| **raw_data_collector.py not integrated** | The rich Greeks/IV data it produces is not fed back into the ML model or strategy. |
| **No margin/capital check** | No pre-trade check for available margin before placing orders. |
| **re-entry doesn't filter on SL** | After a stop-loss, re-entry is immediately attempted on the next bar without a cooldown or minimum gap. |

---

## 📦 Dependencies (`requirements.txt`)

```
kiteconnect>=4.1.0   # Zerodha API client
pandas>=2.0.0        # Data manipulation
numpy>=1.24.0        # Numerical computation
scipy                # Black-Scholes IV solver (used in raw_data_collector.py)
openpyxl             # Excel file generation (used in raw_data_collector.py)
```

> **Not listed but required by `ml_model.py`:**
> - `xgboost` — used directly (`import xgboost as xgb`)
> - `scikit-learn` — used for `TimeSeriesSplit`, `LabelEncoder`, `accuracy_score`
>
> ⚠️ **Add these to `requirements.txt`!**

---

## 🚀 Quick Start (Demo — No Kite Needed)

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Test raw data collector in demo mode (generates synthetic Excel file)
python raw_data_collector.py --demo

# Output: raw_data/NIFTY_24Mar2026.xlsx (22 tabs, ~376 rows each)
```

For the main trading pipeline, a live Kite subscription is required.

---

*Generated: 24 Mar 2026 | Project: Nifty VWAP Straddle Strategy*
