import os
import sys
import pandas as pd
import db
import config
import arjun_model
import krishna_model as ml

# Add parent to path
sys.path.append(os.getcwd())

def battle_of_the_strikes():
    print("\n⚔️ BATTLE OF THE STRIKES — 2025 COMPARISON")
    start_date = "2025-01-01"
    end_date   = "2025-12-31"

    with db.get_conn() as conn:
        feat_df = pd.read_sql(f"SELECT * FROM daily_features WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)
        all_candles = pd.read_sql(f"SELECT trade_date, strike, ts as datetime, straddle_price, vwap, ce_iv, pe_iv, ce_volume, pe_volume FROM straddle_candles WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'", conn)

    model_k = ml.load_model()
    model_a = arjun_model.load_arjun_model()
    days = sorted(feat_df['trade_date'].unique())

    krishna_pnl = 0
    atm_pnl = 0
    
    for trade_date in days:
        day_feat = feat_df[feat_df['trade_date'] == trade_date]
        if day_feat.empty: continue
        
        # 1. Krishna's Choice
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        proba = model_k.predict_proba(X_rows)[:, 1]
        best_idx = proba.argmax()
        k_strike = int(day_feat.iloc[best_idx]["strike"])
        
        # 2. ATM Choice (Closest to strike with highest volume/center)
        # Usually represented by the 6th row in our 11-strike set
        atm_strike = int(day_feat.iloc[5]["strike"]) 

        # Simulate Krishna
        k_candles = all_candles[(all_candles['trade_date'] == trade_date) & (all_candles['strike'] == k_strike)]
        k_trades = arjun_model.simulate_pnl_with_arjun_v2(k_candles, model_a, threshold=0.55, p_target=9999, s_loss=-100)
        krishna_pnl += sum(t['pnl_pts'] for t in k_trades)

        # Simulate ATM
        a_candles = all_candles[(all_candles['trade_date'] == trade_date) & (all_candles['strike'] == atm_strike)]
        a_trades = arjun_model.simulate_pnl_with_arjun_v2(a_candles, model_a, threshold=0.55, p_target=9999, s_loss=-100)
        atm_pnl += sum(t['pnl_pts'] for t in a_trades)

    print(f"\n📊 RESULTS (Total Points 2025):")
    print(f"🔹 Krishna (ML) Strike: {krishna_pnl:.1f} pts")
    print(f"🔸 ATM (Standard) Strike: {atm_pnl:.1f} pts")
    
    win_pct_improvement = ((krishna_pnl - atm_pnl) / abs(atm_pnl)) * 100 if atm_pnl != 0 else 0
    print(f"\n🚀 Krishna Outperformance: {win_pct_improvement:.1f}%")

if __name__ == "__main__":
    battle_of_the_strikes()
