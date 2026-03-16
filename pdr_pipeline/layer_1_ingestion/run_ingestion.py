import logging
import json
import os
from setu_connector import SetuAAConnector
from normalizer import flatten_aa_json

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_layer1_ingestion(session_id: str, output_csv: str = "ingested_transactions.csv"):
    """
    Main orchestrator for Layer 1 Ingestion.
    1. Connects to Setu Account Aggregator to fetch raw JSON data.
    2. Passes JSON through the Normalizer to flatten it into a Pandas DataFrame.
    3. Saves the cleaned data to CSV for Layer 2.
    """
    print("=========================================")
    print("   STARTING LAYER 1: DATA INGESTION      ")
    print("=========================================\n")
    
    # Example logic using our Connector and Normalizer:
    
    connector = SetuAAConnector()
    print(f"[*] Fetching FI data for session ID: {session_id} ...")
    
    try:
        # In actual production, uncomment the line below to call the real API:
        # raw_json_data = connector.fetch_fi_data(session_id)
        
        # We will use mock data locally since we don't have active credentials right now:
        mock_path = os.path.join("mock_data", "mock_aa_data.json")
        with open(mock_path, "r") as f:
            raw_json_data = json.load(f)
            
        print("[+] Raw JSON data fetched successfully.")
        
        print(f"\n[*] Passing data through the Normalizer...")
        df = flatten_aa_json(raw_json_data)
        
        if df.empty:
            print("[!] WARNING: Normalizer returned an empty DataFrame.")
        else:
            print(f"[+] Data Normalized! Total transactions found: {len(df)}")
            print(df.head())
            
            # Save the result to CSV for the next pipeline layer
            df.to_csv(output_csv, index=False)
            print(f"\n[SUCCESS] Layer 1 ingestion complete. Saved to: {output_csv}")
            
    except FileNotFoundError:
        print("[ERROR] Mock data not found in mock_data/mock_aa_data.json")
    except Exception as e:
        print(f"[ERROR] Layer 1 ingestion failed: {e}")

if __name__ == "__main__":
    # Example usage:
    # We pass in the mock SessionID from mock_aa_data.json
    run_layer1_ingestion(session_id="897be985-7a1b-4749-a55d-f13efcd2eff3")
