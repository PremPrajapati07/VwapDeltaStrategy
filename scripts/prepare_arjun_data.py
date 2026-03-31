# ============================================================
#  scripts/prepare_arjun_data.py — Feature Preparation for Model Arjun
# ============================================================
"""
Generates sequential (minute-by-minute) training data for Model Arjun.

Logic:
1. Fetch all 'best strike' rows from daily_features (2025).
2. For each, load its full minute-candle sequence from straddle_candles.
3. Convert timestamps to IST (they are stored as TIMESTAMPTZ/UTC in DB).
4. Filter candles to 9:20 AM–3:00 PM IST window.
5. Calculate dynamic trade features (P&L drift, Greek changes, VWAP dist).
6. Label each minute with 'should_exit' via 30-min oracle look-ahead.
7. Bulk insert into arjun_training_data table.
"""

import os, sys, traceback
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

LOOK_AHEAD_MINUTES  = 30
MIN_CANDLES         = 300   # Skip days with too few candles (partial data)

# Oracle labeling thresholds
DRAWDOWN_EXIT_PCT   = 0.40  # Exit if drawdown is >40% of max profit so far
END_OF_DAY_TIME     = dt.time(14, 30)  # Always label as exit near close
NEGATIVE_PNL_LIMIT  = -30  # Exit if deeply losing (stop loss scenario)


def safe_float(val, default=0.0, clip=1e6):
    """Convert to float safely, clipping extremes."""
    try:
        v = float(val)
        if not np.isfinite(v):
            return default
        return max(-clip, min(clip, v))
    except Exception:
        return default


def to_ist_naive(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to IST naive (no tz)."""
    if series.dt.tz is not None:
        return series.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return series


def process_strike_day(df_candles: pd.DataFrame) -> list:
    """Process one (strike, day) group and return list of DB tuples."""
    rows = []

    df = df_candles.copy()

    # Numeric coerce
    num_cols = ["straddle_price", "vwap", "vwap_gap_pct",
                "ce_delta", "pe_delta", "ce_theta", "pe_theta",
                "ce_iv", "pe_iv", "straddle_volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Timezone → IST naive
    df["ts"] = pd.to_datetime(df["ts"])
    df["ts_ist"] = to_ist_naive(df["ts"])
    df["t"] = df["ts_ist"].dt.time

    # Filter to market hours
    df = df[(df["t"] >= dt.time(9, 20)) & (df["t"] <= dt.time(15, 0))].reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        return []

    # ── Feature Engineering ──────────────────────────────────
    entry_price = safe_float(df.iloc[0]["straddle_price"])
    entry_iv    = safe_float(df.iloc[0]["ce_iv"]) + safe_float(df.iloc[0]["pe_iv"])

    df["pnl_pts"]        = entry_price - df["straddle_price"]
    df["max_pnl_so_far"] = df["pnl_pts"].cummax()
    df["drawdown"]       = df["max_pnl_so_far"] - df["pnl_pts"]
    df["delta_drift"]    = (df["ce_delta"] + df["pe_delta"]).abs()
    df["theta_velocity"] = df["ce_theta"] + df["pe_theta"]
    df["iv_drift"]       = (df["ce_iv"] + df["pe_iv"]) - entry_iv

    # Relative volume (safe — clip at 100x)
    rolling_mean = df["straddle_volume"].rolling(15, min_periods=1).mean()
    df["rel_vol_15m"] = (df["straddle_volume"] / rolling_mean.clip(lower=1)).clip(0, 100)

    # ── Oracle Look-Ahead Labeling (Multi-Condition) ─────────
    should_exit = []
    for i in range(len(df)):
        curr_pnl      = safe_float(df.loc[i, "pnl_pts"])
        curr_drawdown = safe_float(df.loc[i, "drawdown"])
        max_pnl       = safe_float(df.loc[i, "max_pnl_so_far"])
        curr_time     = df.loc[i, "t"]
        future        = df.iloc[i + 1 : i + 1 + LOOK_AHEAD_MINUTES]["pnl_pts"]

        # Condition 1: End of trading day
        if curr_time >= END_OF_DAY_TIME:
            label = True
        # Condition 2: Deep drawdown from peak (losing back >40% of max profit)
        elif max_pnl > 10 and curr_drawdown > max_pnl * DRAWDOWN_EXIT_PCT:
            label = True
        # Condition 3: Deeply negative P&L (stop-loss territory)
        elif curr_pnl < NEGATIVE_PNL_LIMIT:
            label = True
        # Condition 4: Future look-ahead declines from current level
        elif future.empty:
            label = True
        else:
            max_future = safe_float(future.max())
            # Exit if future never recovers even half of current P&L
            label = (max_future < curr_pnl * 0.5) and (curr_pnl > 5)

        should_exit.append(bool(label))

    df["should_exit"] = should_exit

    # ── Build DB Tuples ───────────────────────────────────────
    trade_date = df_candles.iloc[0]["trade_date"]
    strike     = int(df_candles.iloc[0]["strike"])

    for _, row in df.iterrows():
        rows.append((
            trade_date,
            row["ts"],                              # Store original UTC ts in DB
            strike,
            safe_float(row["pnl_pts"]),
            safe_float(row["max_pnl_so_far"]),
            safe_float(row["drawdown"]),
            safe_float(row["vwap_gap_pct"]),
            safe_float(row["delta_drift"]),
            safe_float(row["theta_velocity"]),
            safe_float(row["iv_drift"]),
            safe_float(row["rel_vol_15m"]),
            bool(row["should_exit"]),
        ))
    return rows


def generate_arjun_dataset():
    print("🚀 Extracting Sequential Data for Model Arjun ...")

    with db.get_conn() as conn:
        # Load all best-strike days
        df_best = pd.read_sql("""
            SELECT trade_date, strike
            FROM daily_features
            WHERE is_best = TRUE
              AND trade_date BETWEEN '2025-01-01' AND '2025-12-31'
        """, conn)

        if df_best.empty:
            print("❌ No best-strike data in daily_features. Run --mode train first.")
            return

        print(f"📂 Found {len(df_best)} best-strike sequences. Processing ...")

        arjun_rows = []
        skipped    = 0

        for i, (_, b_row) in enumerate(tqdm(df_best.iterrows(), total=len(df_best))):
            t_date = b_row["trade_date"]
            strike = int(b_row["strike"])

            try:
                df_candles = pd.read_sql(f"""
                    SELECT '{t_date}'::date AS trade_date,
                           {strike}        AS strike,
                           ts, straddle_price, vwap, vwap_gap_pct,
                           ce_delta, pe_delta,
                           ce_theta, pe_theta,
                           ce_iv, pe_iv,
                           straddle_volume
                    FROM straddle_candles
                    WHERE trade_date = '{t_date}' AND strike = {strike}
                    ORDER BY ts
                """, conn)

                result = process_strike_day(df_candles)

                if not result:
                    skipped += 1
                else:
                    arjun_rows.extend(result)

            except Exception as e:
                print(f"\n  ❌ Error on {t_date} Strike {strike}: {e}")
                traceback.print_exc()

        print(f"\n✅ Built {len(arjun_rows)} rows. Skipped {skipped} partial days.")

        if not arjun_rows:
            print("❌ No rows to insert. Exiting.")
            return

        # ── Bulk Insert ──────────────────────────────────────────
        print("📦 Inserting into arjun_training_data ...")
        INSERT_SQL = """
            INSERT INTO arjun_training_data
            (trade_date, ts, strike, pnl_pts, max_pnl_so_far, drawdown,
             vwap_gap_pct, delta_drift, theta_velocity, iv_drift,
             rel_vol_15m, should_exit)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (trade_date, ts, strike) DO NOTHING
        """
        BATCH = 5000
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE arjun_training_data")
            for start in range(0, len(arjun_rows), BATCH):
                batch = arjun_rows[start : start + BATCH]
                try:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, INSERT_SQL, batch, page_size=500)
                except Exception as e:
                    print(f"\n  ❌ Batch insert failed at row {start}: {e}")
                    traceback.print_exc()
                    break

        conn.commit()

    # Verify
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT should_exit, COUNT(*) FROM arjun_training_data GROUP BY should_exit")
            counts = cur.fetchall()

    print("\n📊 Arjun Training Data Label Distribution:")
    total = sum(c[1] for c in counts)
    for label, cnt in counts:
        pct = cnt / total * 100 if total else 0
        tag = "EXIT" if label else "HOLD"
        print(f"   {tag}: {cnt:,} rows  ({pct:.1f}%)")
    print(f"   TOTAL: {total:,} rows")


if __name__ == "__main__":
    generate_arjun_dataset()
