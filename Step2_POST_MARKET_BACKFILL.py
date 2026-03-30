import subprocess
import datetime as dt
import os
import sys

def run():
    print("="*60)
    print("      🔄 STEP 2: POST-MARKET DATA BACKFILL     ")
    print("="*60)
    
    today = dt.date.today().strftime("%Y-%m-%d")
    print(f"\nCollecting all strike data for {today}...\n")
    
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = [python_exe, "main.py", "--mode", "backfill", "--start", today, "--end", today]
    
    try:
        subprocess.run(cmd)
        print("\n✅ Data backfill complete. Ready for Step 3 (Train).")
    except Exception as e:
        print(f"\n❌ Error during backfill: {e}")

if __name__ == "__main__":
    run()
