import json
import logging
import pandas as pd
from normalizer import flatten_aa_json

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_normalization():
    print("--- STARTING NORMALIZER TEST ---")
    mock_file_path = "mock_data/mock_aa_data.json"
    
    print(f"Loading mock data from {mock_file_path}...")
    try:
        with open(mock_file_path, "r") as f:
            mock_data = json.load(f)
            
        print("\nFlattening the JSON payload...")
        df = flatten_aa_json(mock_data)
        
        print("\n--- NORMALIZED DATAFRAME RESULTS ---")
        if df.empty:
            print("The DataFrame is empty! Something went wrong in extraction.")
        else:
            print(df.to_string())
            print(f"\nTotal Records: {len(df)}")
            print("\nDataFrame Types:")
            print(df.dtypes)
            
        print("\n[SUCCESS] Normalizer successfully parsed mock Setu AA Data.")
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find mock data file at {mock_file_path}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_normalization()
