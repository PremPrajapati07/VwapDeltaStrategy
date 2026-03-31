import db

def migrate():
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            print("🕒 Finalizing 'daily_features' table for Relative Greeks...")
            cols = [
                "delta_spread", "theta_ratio", "vega_imbalance"
            ]
            for col in cols:
                try:
                    cur.execute(f"ALTER TABLE daily_features ADD COLUMN {col} NUMERIC;")
                    print(f"  ✅ Added column: {col}")
                except Exception as e:
                    print(f"  ⚠️  Skipping {col}: {e}")
                    conn.rollback()
            conn.commit()
            print("🚀 Migration complete.")

if __name__ == "__main__":
    migrate()
