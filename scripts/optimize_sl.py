import pandas as pd
import os
import io
import sys
from contextlib import redirect_stdout

# Ensure we can import modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main
import config
import db
import re

def run_grid_search():
    sl_range = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    results = []
    
    print("🚀 Starting HIGH-SPEED SL Grid Search (Jan+Feb 2025)...")
    print("-" * 50)
    
    # 1. Ensure DB is ready
    import db
    db.init_schema()
    
    for sl in sl_range:
        print(f"🔍 Testing SL Multiplier: {sl}x ...")
        
        # 2. Redirect stdout to capture the results
        f = io.StringIO()
        with redirect_stdout(f):
             # 3. Call backtest function DIRECTLY
             main.run_backtest(start_date="2025-01-01", end_date="2025-02-28", sl_multiplier=sl)
        
        out = f.getvalue()
        
        # 4. Parse Out Results
        import re
        pnl = 0.0
        win_rate = 0.0
        try:
            pnl_match = re.search(r"Total P&L \(pts\) : ([\s\d\.,-]+)", out)
            if pnl_match:
                pnl_str = pnl_match.group(1).strip().replace(",", "")
                pnl = float(pnl_str)
            
            win_match = re.search(r"Win Days\s+:\s+\d+\s+\(([\d.]+)%\)", out)
            if win_match:
                win_rate = float(win_match.group(1))
            else:
                win_match_alt = re.search(r"(\d+)%", out)
                if win_match_alt:
                    win_rate = float(win_match_alt.group(1))
        except Exception as e:
            print(f"   ⚠️  Parsing error: {e}")
            
        print(f"   ✅ Result: P&L = {pnl:,.1f}, Win Rate = {win_rate}%")
        results.append({"sl": sl, "pnl": pnl, "win_rate": win_rate})
    
    # 5. Final summary
    df = pd.DataFrame(results)
    best = df.loc[df["pnl"].idxmax()]
    print("\n🏆 OPTIMIZATION COMPLETE")
    print("-" * 30)
    print(f"Best SL Multiplier: {best['sl']}x")
    print(f"Max P&L:            {best['pnl']:,.1f}")
    print(f"Win Rate:           {best['win_rate']}%")
    
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/sl_optimization_results.csv", index=False)

if __name__ == "__main__":
    run_grid_search()
