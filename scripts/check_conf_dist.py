import pandas as pd
import db

with db.get_conn() as conn:
    df = pd.read_sql("SELECT confidence FROM krishna_predictions WHERE trade_date BETWEEN '2025-01-01' AND '2025-12-31'", conn)
    print("📈 Krishna Confidence 2025 Distribution:")
    print(df.describe())
    print("\nQuantiles:")
    print(df.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
