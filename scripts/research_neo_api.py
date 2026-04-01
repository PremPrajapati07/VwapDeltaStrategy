import requests
import os

url = "https://raw.githubusercontent.com/Kotak-Neo/kotak-neo-api/main/neo_api_client/api/neo_client.py"
try:
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        # Look for login and place_order
        print("--- KOTAK NEO API STRUCTURE ---")
        lines = content.split("\n")
        
        # Find methods
        for i, line in enumerate(lines):
            if "def login(" in line or "def place_order(" in line:
                print(f"L{i+1}: {line.strip()}")
                # Print a few next lines for arguments
                for j in range(1, 10):
                    if i+j < len(lines):
                        print(f"  {lines[i+j].strip()}")
    else:
        print(f"Error fetching: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
