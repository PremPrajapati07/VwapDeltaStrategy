import krishna_model as ml
import pandas as pd

def check():
    feat_df = ml.build_features_from_db()
    print("--- Days in feat_df ---")
    print(feat_df["trade_date"].unique())
    print("\n--- Row count per day ---")
    print(feat_df.groupby("trade_date").size())
    
    # Check if 2026-03-25 has pnl
    mar25 = feat_df[feat_df["trade_date"].astype(str).str.contains("03-25")]
    print("\n--- March 25 PNL values ---")
    print(mar25[["strike", "pnl"]])

if __name__ == "__main__":
    check()
