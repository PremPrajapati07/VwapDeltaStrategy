import db

def truncate_all():
    print("🗑️  Truncating all tables in PostgreSQL …")
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            # Order matters for foreign keys (if any), but here they are independent
            tables = [
                "sessions",
                "instruments",
                "straddle_candles",
                "daily_features",
                "market_context",
                "trade_log"
            ]
            for table in tables:
                print(f"  - Truncating {table} …")
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
        conn.commit()
    print("✅ Database cleared. Ready for fresh start.")

if __name__ == "__main__":
    truncate_all()
