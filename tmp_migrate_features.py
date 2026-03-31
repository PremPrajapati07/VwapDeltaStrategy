import db

def migrate():
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            print("🕒 Upgrading 'daily_features' table schema...")
            cols = [
                "ce_iv", "pe_iv", "ce_delta", "pe_delta", 
                "ce_theta", "pe_theta", "ce_gamma", "pe_gamma", 
                "ce_vega", "pe_vega"
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
