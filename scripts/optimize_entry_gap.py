import os
import sys
import datetime as dt
import time
import multiprocessing as mp
import pandas as pd
from zoneinfo import ZoneInfo

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import db
import krishna_model as ml
import arjun_model

IST = ZoneInfo("Asia/Kolkata")

def worker_simulate(args):
    """
    Worker function executed in a separate process.
    args: (threshold_pct, daily_data_list)
    """
    threshold_pct, daily_data_list = args
    import config as worker_cfg
    import arjun_model as worker_arjun
    
    # Overwrite the global config for this specific process
    worker_cfg.VWAP_ENTRY_THRESHOLD_PCT = threshold_pct
    
    try:
        arjun_clf = worker_arjun.load_arjun_model()
    except:
        return {"Threshold": f"{threshold_pct}%", "Error": "LoadFail"}

    results = []
    for day_df in daily_data_list:
        if day_df.empty: continue
            
        try:
            raw_td = day_df.iloc[0]["trade_date"]
            trade_date_str = str(raw_td)
        except:
            trade_date_str = "Unknown"
            
        day_trades = worker_arjun.simulate_pnl_with_arjun_v2(
            day_df,
            arjun_model=arjun_clf,
            threshold=worker_cfg.ARJUN_EXIT_THRESHOLD,
            p_target=worker_cfg.DAILY_PROFIT_TARGET,
            s_loss=worker_cfg.DAILY_STOP_LOSS
        )
        
        day_pnl = sum(t["pnl_pts"] for t in day_trades)
        results.append({"pnl": day_pnl, "count": len(day_trades)})
        
    res_df = pd.DataFrame(results)
    if res_df.empty: return {"Threshold": f"{threshold_pct}%", "Total P&L (pts)": 0}
        
    total_pnl = res_df["pnl"].sum()
    win_days  = (res_df["pnl"] > 0).sum()
    max_dd    = res_df["pnl"].cumsum().sub(res_df["pnl"].cumsum().cummax()).min()
    
    return {
        "Threshold": f"{threshold_pct}%",
        "Total P&L (pts)": round(total_pnl, 1),
        "Win Rate (%)": f"{win_days/len(res_df):.1%}",
        "Max DD (pts)": round(max_dd, 1),
        "Total Trades": res_df["count"].sum()
    }

def main():
    print("\n🔍 Nifty VWAP Optimizer (2025)\n")
    with db.get_conn() as conn:
        q_feat = "SELECT * FROM daily_features WHERE trade_date >= '2025-01-01'"
        q_cand = "SELECT * FROM straddle_candles WHERE trade_date >= '2025-01-01'"
        feat_df = pd.read_sql(q_feat, conn)
        candles = pd.read_sql(q_cand + " ORDER BY trade_date, strike, ts", conn)
        
    print(f"✅ Data loaded. Candles: {len(candles):,}")
    
    candles["ts"] = pd.to_datetime(candles["ts"]).dt.tz_localize("UTC").dt.tz_convert(IST)
    numeric_cols = ["vwap", "straddle_price", "ce_close", "pe_close", "atm", "ce_volume", "pe_volume"]
    for col in numeric_cols: 
        if col in candles.columns: candles[col] = pd.to_numeric(candles[col], errors="coerce").fillna(0)
    
    krishna = ml.load_model()
    days = sorted(candles["trade_date"].unique())
    daily_data_list = []
    
    for trade_date in days:
        date_str = str(trade_date)
        day_feat = feat_df[pd.to_datetime(feat_df["trade_date"]).dt.strftime('%Y-%m-%d') == date_str]
        if day_feat.empty: continue
        X_rows = day_feat[config.ML_FEATURES].fillna(0)
        best_strike = int(day_feat.iloc[krishna.predict_proba(X_rows)[:, 1].argmax()]["strike"])
        day_candles = candles[(candles["trade_date"] == trade_date) & (candles["strike"] == best_strike)].copy()
        if not day_candles.empty:
            day_candles = day_candles.rename(columns={"ts": "datetime"})
            daily_data_list.append(day_candles)

    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    pool_args = [(float(t), daily_data_list) for t in thresholds]
    
    with mp.Pool(processes=max(1, os.cpu_count() - 2)) as pool:
        results = pool.map(worker_simulate, pool_args)
        
    print("\n" + "=" * 80)
    print("  🏆 VWAP ENTRY GAP OPTIMIZATION RESULTS (2025)")
    print("=" * 80)
    print(pd.DataFrame(results).sort_values("Total P&L (pts)", ascending=False).to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
