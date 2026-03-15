import time
import json
import logging
from datetime import datetime, timedelta, timezone
from setu_connector import SetuAAConnector
from normalizer import flatten_aa_json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layer_2_feature_engine import FeatureStoreMSME

# Setup basic logging to see the output clearly
logging.basicConfig(level=logging.INFO, format='%(message)s')

def orchestrate_live_pipeline():
    """
    End-to-end orchestrator that initiates Setu Consent -> Waits for User Approval -> 
    Fetches Live 365 JSON FI Stream -> Normalizes DataFrame -> Generates XGBoost Feature Vector.
    """
    print("\n=======================================================")
    print("      LIVE PIPELINE ORCHESTRATOR: CONSENT TO FEATURES  ")
    print("=======================================================\n")
    
    # 1. Initialize Setu connector
    connector = SetuAAConnector()

    # 2. Setup 1-Year maximum dynamic fetch range
    now_utc = datetime.now(timezone.utc)
    fetch_start = (now_utc - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    data_range_payload = {
        "from": fetch_start,
        "to": fetch_end
    }

    # 3. Define the comprehensive Consent Payload for V2
    consent_payload = {
        "vua": "",
        "consentDuration": {
            "unit": "MONTH",
            "value": 1
        },
        "consentTypes": ["TRANSACTIONS", "PROFILE", "SUMMARY"],
        "fiTypes": ["DEPOSIT"],
        "dataRange": data_range_payload,
        "purpose": {
            "code": "101",
            "refUri": "https://api.rebit.org.in/aa/purpose/101.xml",
            "text": "Credit Scoring & Wealth analysis",
            "category": {"type": "string"}
        },
        "fetchType": "PERIODIC",
        "frequency": {
            "unit": "MONTH",
            "value": 1
        },
        "dataLife": {
            "unit": "MONTH",
            "value": 1
        },
        "redirectUrl": "https://your-app-dashboard.com/success"
    }

    print(f"[*] Requesting LIVE 'DEPOSIT' mapping from: {fetch_start} to {fetch_end}")

    # =========================================================
    # TASK 1: THE CONNECTION LOGIC (SETU API)
    # =========================================================
    try:
        # A) Request Consent Approval URL
        consent_resp = connector.create_consent_request(consent_payload)
        consent_id = consent_resp.get("id")
        consent_url = consent_resp.get("url")
        
        if not consent_id or not consent_url:
            print("[ERROR] Invalid Consent Payload formatting returned from Setu Sandbox. Missing ID/URL.")
            return

        print("\n[SUCCESS] CONSENT INITIATED")
        print("\n---------------------------------------------------------")
        print(f"[*] ACTION REQUIRED (Sandbox Testing):")
        print(f"1. Open this URL natively in your browser: \n>> {consent_url}")
        print(f"2. Approve the consent prompt for consent ID: {consent_id}")
        print("---------------------------------------------------------\n")
        
        # B) Polling Loop (Wait for 'ACTIVE' status)
        print("[*] Polling consent status every 10 seconds. Waiting for User Approval...")
        max_attempts = 12 # Timeout after 2 mins
        is_approved = False
        
        for attempt in range(max_attempts):
            status_resp = connector.get_consent_status(consent_id)
            status = status_resp.get("status", "UNKNOWN")
            
            print(f"  -> Attempt {attempt+1}/{max_attempts}: Status is [{status}]")
            
            if status == "ACTIVE":
                is_approved = True
                print("\n[SUCCESS] User has APPROVED the datastream flow!")
                break
            elif status in ["REJECTED", "REVOKED"]:
                print(f"\n[ERROR] User {status} the consent request. Aborting pipeline.")
                return
                
            time.sleep(10)
            
        if not is_approved:
            print("\n[TIMEOUT] User did not approve within the 2-minute polling window.")
            return

        # C) Session Generation & FI Hook
        print("\n[*] Requesting Data Session Generation...")
        session_resp = connector.create_data_session(consent_id, data_range_payload)
        session_id = session_resp.get("id")
        
        if not session_id:
            print("[ERROR] Failed to map an active data session from the consent ID.")
            return
            
        print(f"  -> Data Session '{session_id}' activated.")
        
        print("\n[*] Pulling comprehensive FI Accounts JSON Payload...")
        # Since Data session creation on Sandbox can take a few seconds async to compile the JSON
        # we attach a brief throttle hook to let it generate
        time.sleep(5) 
        live_json_payload = connector.fetch_fi_data(session_id)
        
        print("[SUCCESS] Live Financial Information (FI) stream retrieved securely.")

        # =========================================================
        # TASK 2: THE INTEGRATION BRIDGE (JSON -> Normalized Matrix)
        # =========================================================
        print("\n[*] Bridging into the Layer 1 Pandas Normalizer...")
        df = flatten_aa_json(live_json_payload)
        
        if df.empty:
            print("[!] WARNING: The live stream returned an empty matrix/no transactions.")
            return
            
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
        print(f"[+] Bridge Established. Flattened dataframe bounds '{len(df)}' row vectors.")
        
        # =========================================================
        # TASK 3: MAXIMUM TIMELINE VERIFICATION
        # =========================================================
        actual_min_date = df['Date'].min()
        actual_max_date = df['Date'].max()
        
        # We calculate the delta
        time_diff = (actual_max_date - actual_min_date).days
        
        print("\n---------------------------------------------------------")
        print("[*] TIMELINE AUDIT & INTEGRITY CHECK")
        print(f"  -> Requested Window: {fetch_start} TO {fetch_end}")
        print(f"  -> Actual Data Received: {actual_min_date} TO {actual_max_date}")
        print(f"  -> Total Data Day-Span: {time_diff} days")
        
        if time_diff >= 350: # Providing 2 weeks tolerance on the dataset
            print("  -> [PASS] Verified ~365 days temporal history present.")
        else:
            print("  -> [WARNING] Insufficient temporal footprint retrieved from AA Sandbox!")
        print("---------------------------------------------------------\n")

        # =========================================================
        # TASK 4: LAYER 2 XGBOOST FEATURE GENERATION
        # =========================================================
        print("[*] Forwarding Flattened Matrix to Layer 2 Feature Engine MSME Vector...")
        
        # Mocking the UI Form Profile elements for MSME
        ui_data_msme = {
            "gst_declared_revenue": 5000000,
            "telecom_number_vintage_days": 1800,
            "academic_background_tier": 1,
            "location_type": "urban",
            "earning_members": 2, 
            "total_members": 5,
        }
        
        fs = FeatureStoreMSME(df, ui_data_msme)
        xgb_vector = fs.generate_feature_vector()
        
        print("\n[SUCCESS] PDR PIPELINE EXECUTION FINISHED")
        print("Exporting XGBoost Layer Inference Vector:\n")
        print(json.dumps(xgb_vector, indent=4))
        
        # Finally, save artifacts
        csv_path = "live_ingested_transactions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Artifacts archived to {csv_path}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution fractured: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("Underlying Exception Traces:")
            print(e.response.text)


if __name__ == "__main__":
    orchestrate_live_pipeline()
