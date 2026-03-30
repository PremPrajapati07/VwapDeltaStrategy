import subprocess
import os
import sys

def run():
    print("="*60)
    print("      📈 STEP 4: VIEW STRATEGY REPORT (BACKTEST)     ")
    print("="*60)
    print("\nRunning backtest on all historical data...\n")
    
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = [python_exe, "main.py", "--mode", "backtest"]
    
    try:
        subprocess.run(cmd)
        print("\n✅ Report generated. Check logs/backtest_report.csv for details.")
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")

if __name__ == "__main__":
    run()
