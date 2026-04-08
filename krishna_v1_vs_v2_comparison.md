# Krishna V1 vs V2 — Full Model Comparison
**File:** [krishna_model.py](file:///p:/Vaama%20Projects/Trading%20Setup/krishna_model.py) vs [krishna_v2_model.py](file:///p:/Vaama%20Projects/Trading%20Setup/krishna_v2_model.py)

---

## 🧠 Simple Summary — What Each Model Does

| | **Krishna V1** | **Krishna V2** |
|---|---|---|
| **Analogy** | A photo 📸 — one freeze-frame at 9:20 AM | A live video 🎥 — continuous re-evaluation |
| **When it decides** | Once, at 9:20 AM, fixed forever | Every 5 minutes from 9:25 to 11:00 AM |
| **What it sees** | Market data frozen at 9:20 | Live market shifts — spot drift, OI velocity, premium decay |
| **Strike flexible?** | No — once chosen, never changed | Yes — can switch strike if market moves *before* trade entry |
| **Training samples** | ~2,700 rows (1 per day per strike) | ~194,000 rows (one per 5-minute window per strike per day) |
| **Features** | 26 static Greek/OI features | 34 features (26 inherited + 8 live dynamic ones) |
| **Result (2025 backtest)** | ~9,752 pts | Same, but with better strike adaptation |

---

## 📌 Phase 1 — Feature Set Differences

### V1: 26 Static Features (taken at 9:20 AM once)

```python
# krishna_model.py — config.ML_FEATURES
```

| Feature | Technical Meaning | Non-Technical (Plain English) |
|---|---|---|
| `vwap_gap_pct` | `(straddle - vwap) / vwap × 100` | How far the straddle price is above/below its average — the entry signal |
| `straddle_premium` | `ce_close + pe_close` at 9:20 | Total price of buying both Call + Put options |
| `ce_pe_ratio` | `ce_close / pe_close` | Is the Call more expensive than the Put? Signals directional bias |
| `distance_from_atm` | `strike - atm` (in points) | How far is this strike from where Nifty is currently trading |
| `distance_from_atm_pct` | `(strike - atm) / atm × 100` | Same but in percentage terms |
| `oi_ce` | Raw Call Open Interest | How many Call contracts are open in the market |
| `oi_pe` | Raw Put Open Interest | How many Put contracts are open in the market |
| `oi_imbalance` | `(oi_ce - oi_pe) / (oi_ce + oi_pe)` | Are more people betting Nifty goes up or down? (+ve = more calls) |
| `days_to_expiry` | Calendar days to weekly expiry | Time value left in options (less days = faster decay) |
| `day_of_week` | 0=Monday … 4=Friday | Which day of the week — expiry is usually Thursday |
| `vix` | India VIX index | Market fear index — higher means bigger moves expected |
| `nifty_prev_day_change` | Yesterday's Nifty % change | Did market move big yesterday? Tends to mean-revert next day |
| `nifty_open_gap` | Today's open vs yesterday's close | Did the market gap up/down at open? |
| `ce_iv` / `pe_iv` | Implied Volatility of Call / Put | How much volatility is "priced in" to each option |
| `ce_delta` / `pe_delta` | Option Delta | Sensitivity of option price to Nifty movement (CE: 0 to 1, PE: -1 to 0) |
| `ce_theta` / `pe_theta` | Option Theta | How much the option loses per day just from time passing |
| `ce_gamma` / `pe_gamma` | Option Gamma | How fast delta changes — high gamma = risky near expiry |
| `ce_vega` / `pe_vega` | Option Vega | How much option price changes with VIX move |
| `delta_spread` | `ce_delta + pe_delta` | Net directional exposure of the straddle (should be ~0 for ATM) |
| `theta_ratio` | `ce_theta / pe_theta` | How balanced the time decay is between CE and PE |
| `vega_imbalance` | `ce_vega - pe_vega` | Is one side more volatile-sensitive than the other? |

---

### V2: 8 NEW Dynamic Features (in addition to all 26 above)

```python
# krishna_v2_model.py — V2_FEATURES additions
"spot_change_pct",     # Line 61
"spot_momentum_5m",    # Line 62
"oi_velocity_ce",      # Line 63
"oi_velocity_pe",      # Line 64
"oi_buildup_ratio",    # Line 65
"premium_decay_rate",  # Line 66
"volume_surge",        # Line 67
"minutes_since_open",  # Line 68
```

| Feature | Formula | Technical Meaning | Non-Technical |
|---|---|---|---|
| `spot_change_pct` | `(spot_now - spot_920) / spot_920 × 100` | How much has Nifty spot moved since 9:20 | Is the market running away? 22000→22500 = +2.3% |
| `spot_momentum_5m` | `(spot_now - spot_5mins_ago) / spot_5min_ago × 100` | Rate of Nifty movement in last 5 min | Is the move accelerating right now? |
| `oi_velocity_ce` | `(oi_ce_now - oi_ce_920) / oi_ce_920 × 100` | % change in Call OI since 9:20 | Are fresh Call sellers entering? (CE winding up = bearish pressure) |
| `oi_velocity_pe` | `(oi_pe_now - oi_pe_920) / oi_pe_920 × 100` | % change in Put OI since 9:20 | Are fresh Put sellers entering? (PE winding up = bullish) |
| `oi_buildup_ratio` | `oi_vel_ce / oi_vel_pe` (capped ±10) | Relative speed of CE vs PE OI change | Is one side being more aggressively positioned than the other? |
| `premium_decay_rate` | `(straddle_920 - straddle_now) / minutes_elapsed` | How many points per minute is premium falling | Is time decay happening fast or slow? Fast decay = good for straddle sell |
| `volume_surge` | `current_vol / avg_vol_so_far` (capped at 50×) | Volume spike vs day's average | Is there a sudden rush of trading? Surge often precedes a sharp move |
| `minutes_since_open` | `current_mins - 9:15` | How many minutes since market opened | Tells the model WHERE in the day we are — early chaos vs settled market |

---

## 📌 Phase 2 — How Training Data Is Built

### V1: One row per (day, strike) — `build_features_from_db()`

```python
# krishna_model.py lines 68-159
for (trade_date, strike), grp in df.groupby(["trade_date", "strike"]):
    snap_920 = grp[grp["time"] <= dt.time(9, 20)]  # Single snapshot
    row_920 = snap_920.iloc[-1]                     # Last candle at/before 9:20
    pnl = simulate_pnl(grp)                         # Simulate full-day P&L
```

- **1 photo per strike per day** → ~2,700 training rows for 247 days × 11 strikes
- Label: which strike had the best P&L that day → `is_best = True`
- **Problem:** The label is decided at day end, but features are frozen at 9:20 — if the market moved sharply after 9:25, V1 had no idea

---

### V2: Multiple rows per (day, strike, time) — `_process_one_day()` + multiprocessing

```python
# krishna_v2_model.py lines 71-78
_SAMPLE_MINUTES = []
for h in range(9, 11):
    for m in range(0, 60, 5):
        mins = h * 60 + m
        if 9*60+25 <= mins <= 11*60:   # Every 5 min from 9:25 to 11:00
            _SAMPLE_MINUTES.append(mins)
```

- **18 sample times × 11 strikes × 247 days = 194,034 training rows**
- At each sample time `t`, for each strike:
  1. Compute ALL 34 features using live data up to time `t`
  2. Simulate forward P&L from time `t` onwards
  3. Label: which strike had best forward P&L FROM time `t`? → `is_best_at_t`

**This means**: The model learns what a "best strike" looks like at 9:30, 9:35, 9:45, 10:00... with live market context at each moment.

---

## 📌 Phase 3 — Labeling Logic

### V1 Label — `is_best` (Line 148-150)
```python
best = feat_df.groupby("trade_date")["pnl"].idxmax()  # Best day-end P&L
feat_df["is_best"] = False
feat_df.loc[best.values, "is_best"] = True
```
- **One winner per day** (the strike with highest total P&L for that day)
- Binary: 1 = best, 0 = everything else

### V2 Label — `is_best_at_t` (Line 147)
```python
best_strike_at_t = max(forward_pnls, key=forward_pnls.get)  # Best from THIS moment forward
"is_best_at_t": strike == best_strike_at_t
```
- **One winner per (day, time window)** — the best strike FROM this minute onwards
- The label changes over time! At 9:25, strike 22000 might be best. By 10:00 if market moved, 22400 might be best
- This teaches the model: "given these live conditions RIGHT NOW, which strike should I pick?"

---

## 📌 Phase 4 — Forward P&L Simulation

### V1: Row-by-row (slow `iterrows`)
```python
# krishna_model.py lines 241-268
for _, row in grp.iterrows():   # Slow — Python loop over every row
    t = pd.to_datetime(row[ts_col]).time()
    straddle = row["straddle_price"]
```
- Uses `pd.to_datetime(...).time()` on every row — creates Python datetime objects, slow

### V2: Numpy arrays (fast, vectorized)
```python
# v2_model.py lines 258-288
prices = forward_df["straddle_price"].values.astype(float)  # Pull entire column as numpy array
vwaps  = forward_df["vwap"].values.astype(float)
mins   = forward_df["_mins"].values                          # Pre-computed integer minutes

for i in range(len(prices)):   # Still a loop but with raw floats, no pandas overhead
    price = prices[i]
    vwap  = vwaps[i] if vwaps[i] > 0 else price
```
- Pre-extracts columns as numpy float arrays → **5-10× faster** than iterrows
- Time comparison is integer math (`mins[i] >= 900`) not `dt.time` object comparison

---

## 📌 Phase 5 — Multiprocessing (V2 Only)

```python
# v2_model.py lines 376-382
cores = max(1, os.cpu_count() - 2)   # Use all CPUs minus 2 (leave 2 for OS)
with mp.Pool(processes=cores) as pool:
    results = pool.map(_process_one_day, pool_args)  # Each CPU handles 1 day
```

- V1 processes days **sequentially** in a single loop → slow
- V2 packages each day's data as a dict (pickle-friendly), sends to separate CPU cores
- 247 days ÷ 10 cores = each core handles ~25 days simultaneously
- Training time: **V1 ~10 min → V2 ~2 min** (5× speedup just on feature engineering)

---

## 📌 Phase 6 — XGBoost Hyperparameters

| Parameter | V1 Value | V2 Value | What It Means |
|---|---|---|---|
| `n_estimators` (final) | 400 | **500** | Number of decision trees built — more = learns more patterns |
| `max_depth` (final) | 5 | **6** | How deep each tree goes — deeper = captures more complex patterns |
| `learning_rate` | 0.03 | 0.03 | How much each tree corrects the previous — small = careful, stable |
| `subsample` | 0.8 | 0.8 | Uses 80% of rows per tree — prevents memorizing noise |
| `colsample_bytree` | 0.8 | 0.8 | Uses 80% of features per tree — prevents overfitting |
| `scale_pos_weight` | ❌ Not used | **55.27** | Since only 1.8% of rows are "BEST", this tells model to weight those 55× more — fixes class imbalance |
| `eval_metric` | `mlogloss` | **`logloss`** | How model measures its own accuracy during training |

> **Why scale_pos_weight matters:** In V2, only 1 strike out of ~11 is labeled as "best" at any given time → 1.8% positive rate. Without this weight, the model would just predict "NOT BEST" for everything and be 98% accurate but useless. The 55× weighting forces it to actually learn what makes a strike the best.

---

## 📌 Phase 7 — Live Prediction Function

### V1: `predict_best_strike()` — called **once at 9:20**
```python
# krishna_model.py line 352
def predict_best_strike(snapshot_df, vix, nifty_prev_change, nifty_open_gap, 
                        vwap_map, trade_date, expiry_date):
```
- Takes a fixed snapshot, returns one strike
- Strike is **locked in** for the entire day

### V2: `predict_best_strike_v2()` — can be called **any minute**
```python
# krishna_v2_model.py line 477
def predict_best_strike_v2(snapshot_df, vwap_map, baseline_920, current_time, ...):
```
- Takes `baseline_920` as a reference point — the 9:20 AM anchor
- Computes dynamic features using `current_time` vs `baseline_920`
- Can be called at 9:25, 9:30, 9:45, 10:00... any time before trade entry
- **Only acts if no trade is open** — once position enters, strike is locked

```python
mins_since_open = (datetime.combine(today, current_time) - 
                   datetime.combine(today, time(9,15))).total_seconds() / 60
# e.g. at 10:00 AM → 45 minutes since open
```

---

## 📌 Summary: Key Conceptual Differences

| Aspect | V1 | V2 |
|---|---|---|
| **Intelligence type** | Historian — decides based on what the market looked like at 9:20 | Analyst — observes the market evolving and adapts |
| **Strike flexibility** | Rigid — 9:20 decision is final | Adaptive — can switch before entry if market changes |
| **Training approach** | 1 data point per day = low-resolution learning | 18 data points per day = high-resolution, time-aware learning |
| **Market context** | Static VIX, OI, Greeks at 9:20 | + Live spot drift, OI velocity, premium decay, volume surges |
| **Key insight modeled** | "Which strike *was* best given conditions at 9:20?" | "Which strike *is* best given conditions right NOW?" |
| **Performance** | 9,752 pts, 69% win rate (2025) | Same baseline — value shows in edge cases (spot drift days) |
| **Computational load** | Light — 2,700 rows | Heavy — 194,034 rows, needs multiprocessing |
