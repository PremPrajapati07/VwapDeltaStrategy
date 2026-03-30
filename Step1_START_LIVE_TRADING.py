import subprocess
import sys
import os

def run():
    print("="*60)
    print("      🚀 STEP 1: STARTING LIVE TRADING (PAPER MODE)     ")
    print("="*60)
    print("\nStarting the VWAP Straddle scanner...\n")
    
    # Use the local venv python
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = [python_exe, "main.py", "--mode", "live"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 Trading stopped by user.")

if __name__ == "__main__":
    run()
