import subprocess
import os
import sys

def run():
    print("="*60)
    print("      🧠 STEP 3: RETRAIN ML MODEL     ")
    print("="*60)
    print("\nTraining the model with the latest data...\n")
    
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = [python_exe, "main.py", "--mode", "train"]
    
    try:
        subprocess.run(cmd)
        print("\n✅ Model trained! Ready for tomorrow's run.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    run()
