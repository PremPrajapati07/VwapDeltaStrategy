import db
import sys

def main():
    date_to_clear = "2026-03-30"
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trade_log WHERE trade_date = %s", (date_to_clear,))
                cur.execute("DELETE FROM krishna_predictions WHERE trade_date = %s", (date_to_clear,))
        print(f"✅ Cleaned up data for {date_to_clear} from trade_log and krishna_predictions.")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
