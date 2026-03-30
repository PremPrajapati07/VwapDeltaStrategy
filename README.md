# Nifty VWAP Straddle Strategy — ML-Powered Strike Selector

## Strategy Summary
- **Instrument:** Nifty 50 weekly options (Thursday expiry)
- **Setup:** At 9:15 scan 11 strikes (ATM ± 5 × 50 pts)
- **Strike Selection:** ML model picks the single best strike at 9:20 AM
- **Entry:** Sell straddle when straddle price drops **below** its VWAP
- **Exit:** Buy back when straddle price rises **above** VWAP
- **Stop Loss:** 2× premium collected on either leg
- **Hard Exit:** 3:00 PM force-close all positions
- **Re-entry:** Allowed; ML re-evaluates strike on each re-entry

---

## File Structure
```
nifty_straddle/
├── config.py          # All parameters & API credentials
├── data_collector.py  # Kite API integration, VWAP computation, DB storage
├── ml_model.py        # Feature engineering, XGBoost training, prediction
├── live_trader.py     # Real-time VWAP monitor, order execution
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
├── data/              # SQLite database (auto-created)
├── models/            # Saved ML model (auto-created)
└── logs/              # Trade logs (auto-created)
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials
Edit `config.py`:
```python
KITE_API_KEY    = "your_api_key"
KITE_API_SECRET = "your_api_secret"
```
Get these from: https://developers.kite.trade/

### 3. Login
```bash
python main.py --mode login
```
Opens Zerodha login in browser. Paste the `request_token` from the redirect URL.

---

## Workflow

### Step 1 — Backfill Historical Data
```bash
python main.py --mode backfill --start 2023-01-01 --end 2024-12-31
```
Fetches 1-minute CE+PE OHLCV for all 11 strikes on every past Thursday.
Takes ~2–4 hours for 2 years of data (Kite rate limits).

### Step 2 — Train the ML Model
```bash
python main.py --mode train
```
- Simulates VWAP crossover P&L for every strike on every historical Thursday
- Labels each day with the best-performing strike
- Trains XGBoost classifier with TimeSeriesSplit cross-validation
- Prints feature importance and CV accuracy

### Step 3 — Backtest
```bash
python main.py --mode backtest
```
Simulates the full strategy (ML strike selection + VWAP trades) on historical data.
Outputs: Win rate, Total P&L, Sharpe ratio, Strike selection accuracy, Max drawdown.

### Step 4 — Live Paper Trade
```bash
python main.py --mode live --lots 1
```
Runs in paper trade mode (no real orders). Prints all signals and P&L.

### Step 5 — Live Real Trade (Caution!)
```bash
python main.py --mode live --lots 1 --real
```
Places actual orders via Kite. Requires confirmation prompt.

---

## ML Features Used for Strike Selection

| Feature | Description |
|---|---|
| `vwap_gap_pct` | (Straddle - VWAP) / VWAP at 9:20 — how close to entry |
| `straddle_premium` | Total CE+PE premium — theta decay potential |
| `ce_pe_ratio` | CE/PE skew — directional bias |
| `distance_from_atm` | Strike distance from spot in points |
| `distance_from_atm_pct` | Same as % — relative moneyness |
| `oi_ce / oi_pe` | Open Interest per leg |
| `oi_imbalance` | OI skew between CE and PE |
| `days_to_expiry` | Calendar days to Thursday |
| `day_of_week` | Mon=0 … Thu=3 |
| `vix` | India VIX — implied volatility regime |
| `nifty_prev_day_change` | Previous day % move |
| `nifty_open_gap` | Today's open gap % |

---

## Important Notes

1. **Kite API Rate Limit:** ~3 requests/second. Backfill auto-throttles.
2. **Historical Data:** Kite provides up to 60 days of 1-min data on free plan; 400 days on paid.
3. **Lot Size:** Nifty lot size is 50 units. Adjust `--lots` accordingly.
4. **SEBI Margins:** Ensure sufficient SPAN + exposure margin for options selling.
5. **Paper Trade First:** Always validate with paper trading before going live.

---

## Disclaimer
This software is for educational purposes only. Options selling involves unlimited risk.
Always consult a SEBI-registered advisor before trading.
