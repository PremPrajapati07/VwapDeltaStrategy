import db

def migrate():
    print("🚀 Starting database migration ...")
    
    # New columns to add
    cols = [
        ("synthetic_spot", "NUMERIC(10,2)"),
        ("ce_iv", "NUMERIC(10,4)"),
        ("pe_iv", "NUMERIC(10,4)"),
        ("ce_delta", "NUMERIC(10,4)"),
        ("ce_theta", "NUMERIC(10,4)"),
        ("ce_gamma", "NUMERIC(10,6)"),
        ("ce_vega", "NUMERIC(10,4)"),
        ("pe_delta", "NUMERIC(10,4)"),
        ("pe_theta", "NUMERIC(10,4)"),
        ("pe_gamma", "NUMERIC(10,6)"),
        ("pe_vega", "NUMERIC(10,4)"),
        ("nifty_open", "NUMERIC(10,2)")
    ]
    
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for col_name, col_type in cols:
                print(f"  Adding column {col_name} ...")
                try:
                    if col_name == "nifty_open":
                        cur.execute(f"ALTER TABLE market_context ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                    else:
                        # Add to both tables to be safe, or just where they are needed
                        cur.execute(f"ALTER TABLE straddle_candles ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                        cur.execute(f"ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    print(f"  ⚠️  Error adding {col_name}: {e}")
            
    print("✅ Migration complete.")

if __name__ == "__main__":
    migrate()
