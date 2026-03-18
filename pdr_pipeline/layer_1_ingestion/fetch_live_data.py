import time
import json
import logging
import pandas as pd
import requests
import argparse
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from setu_connector import SetuAAConnector
from normalizer import flatten_aa_json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layer_2_feature_engine import FeatureStoreMSME
from layer_3_inference_engine import InferenceEngine

# Setup basic logging to see the output clearly
logging.basicConfig(level=logging.INFO, format='%(message)s')

def _compute_date_range(days: int, *, fixed_end_iso: str | None = None) -> dict:
    """
    Builds a Setu AA dataRange payload.
    If `fixed_end_iso` is provided, we keep `to` fixed (critical: session FIDataRange must be within consent FIDataRange).
    """
    if fixed_end_iso:
        end_dt = datetime.strptime(fixed_end_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    fetch_start = (end_dt - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"from": fetch_start, "to": fetch_end}


def orchestrate_live_pipeline(mobile: str | None = None, days: int = 30):
    """
    End-to-end orchestrator that initiates Setu Consent -> Waits for User Approval -> 
    Fetches Live 365 JSON FI Stream -> Normalizes DataFrame -> Generates XGBoost Feature Vector.
    """
    print("\n=======================================================")
    print("      LIVE PIPELINE ORCHESTRATOR: CONSENT TO FEATURES  ")
    print("=======================================================\n")
    
    # 1. Initialize Setu connector
    connector = SetuAAConnector()
    api_base_url = connector.base_url

    # 2. Setup dynamic fetch range.
    # NOTE: Sandbox FIPs are often flaky for large windows; default to 30d and auto-retry smaller windows on errors.
    # Freeze the consent window end-time so later session calls never exceed consent FIDataRange.
    consent_end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data_range_payload = _compute_date_range(days, fixed_end_iso=consent_end_iso)
    fetch_start = data_range_payload["from"]
    fetch_end = data_range_payload["to"]
    print("\n---------------------------------------------------------")
    print("SETU UAT SANDBOX: LIVE DATA INTEGRATION")
    if not mobile:
        mobile = os.environ.get("SETU_MOBILE")

    if mobile:
        user_mobile = str(mobile).strip()
        print(f"[*] Using mobile from args/env: {user_mobile}")
    else:
        # Only prompt if we're truly interactive
        if sys.stdin is None or not sys.stdin.isatty():
            user_mobile = ""
        else:
            user_mobile = input("[?] Enter your 10-digit mobile number (to receive the live OTP): ").strip()

    if not user_mobile or len(user_mobile) != 10:
        print("[!] Invalid mobile number. Defaulting to dummy 9999999999.")
        user_mobile = "9999999999"
    vua = f"{user_mobile}@setu"
    
    # 3. Define the comprehensive Consent Payload for V2
    consent_payload = {
        "vua": vua,
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
            "text": "Wealth management service",
            "category": {"type": "string"}
        },
        "fetchType": "PERIODIC",
        # IMPORTANT: This frequency governs how often FI requests can be made per consent.
        # Sandbox debugging often requires retries; keep it permissive.
        "frequency": {"unit": "HOUR", "value": 1},
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

        # IMPORTANT: The consent webview host can differ from the API host.
        # Your credentials are typically environment-scoped, so we keep API calls
        # pinned to the original connector base URL (usually fiu-sandbox).
        consent_host_base = None
        try:
            parsed = urlparse(consent_url)
            if parsed.scheme and parsed.netloc:
                consent_host_base = f"{parsed.scheme}://{parsed.netloc}"
                if consent_host_base != api_base_url:
                    print(f"[INFO] Consent webview host: {consent_host_base}")
                    print(f"[INFO] Keeping FIU API host for calls: {api_base_url}")
        except Exception:
            consent_host_base = None
        
        # B) Polling Loop (Wait for 'ACTIVE' status)
        print("[*] Polling consent status every 10 seconds. Waiting for User Approval...")
        max_attempts = 12 # Timeout after 2 mins
        is_approved = False
        
        for attempt in range(max_attempts):
            # Prefer polling on the API base URL, but fall back to the webview host if needed.
            connector.set_base_url(api_base_url)
            try:
                status_resp = connector.get_consent_status(consent_id)
            except requests.exceptions.RequestException:
                if consent_host_base:
                    connector.set_base_url(consent_host_base)
                    status_resp = connector.get_consent_status(consent_id)
                else:
                    raise
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
        connector.set_base_url(api_base_url)
        session_resp = connector.create_data_session(consent_id, data_range_payload)
        session_id = session_resp.get("id") or session_resp.get("sessionId")

        if not session_id:
            print("[ERROR] Failed to map an active data session from the consent ID.")
            return

        print(f"  -> Data Session '{session_id}' activated.")

        # Poll status; if Setu reports the FIP returned invalid response, this is a sandbox/FIP issue.
        # Creating multiple sessions under the same consent can violate consent frequency, so we don't auto-retry here.
        print("[*] Polling Data Session status to ensure FIP has delivered data...")
        is_ready = False
        for attempt in range(10):
            try:
                status_resp = connector.get_session_status(session_id)
            except requests.exceptions.HTTPError as e:
                body = ""
                try:
                    if e.response is not None:
                        body = e.response.text or ""
                except Exception:
                    body = ""
                if "FIP returned an invalid response" in body:
                    print("\n[ERROR] Sandbox FIP returned an invalid response.")
                    print("       This is a provider-side sandbox issue. Try:")
                    print("       - selecting a different bank/account in the consent screen")
                    print("       - re-running with a smaller range: --days 7")
                    print("       - trying again later (sandbox instability)")
                    return
                raise

            sess_status = status_resp.get("status")
            print(f"  -> Attempt {attempt+1}/10: Data Session Status is [{sess_status}]")
            if sess_status in ["COMPLETED", "PARTIAL", "READY"]:
                is_ready = True
                break
            elif sess_status in ["FAILED", "EXPIRED"]:
                print(f"[ERROR] Session failed with status {sess_status}")
                return
            time.sleep(5)

        if not is_ready:
            print("[ERROR] Timed out waiting for data session to become ready.")
            return
        
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
        
        fs = FeatureStoreMSME(df, ui_data_msme, {})
        xgb_vector = fs.generate_feature_vector()
        
        print("\n[SUCCESS] PDR PIPELINE EXECUTION FINISHED")
        print("Exporting XGBoost Layer Inference Vector:\n")
        print(json.dumps(xgb_vector, indent=4))
        
        # Finally, save artifacts
        csv_path = "live_ingested_transactions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Artifacts archived to {csv_path}")

        # =========================================================
        # TASK 5: LAYER 3 INFERENCE CLASSIFICATION
        # =========================================================
        print("\n=======================================================")
        print("                 LIVE INFERENCE DECISION               ")
        print("=======================================================")
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'pdr_xgb_realworld.json')
        if os.path.exists(model_path):
            engine = InferenceEngine(model_path=model_path)
            prediction = engine.predict(xgb_vector)
            
            print(f"\nFinal Risk Decision:  {prediction['decision']}")
            print(f"Computed Risk Score:  {prediction['risk_score']:.4f}")
            if prediction.get('policy_overrides'):
                print(f"Policy Overrides:     {', '.join(prediction['policy_overrides'])}")
        else:
            print(f"\n[!] Model not found at {model_path}. Please train it first.")
        print("=======================================================\n")

    except requests.exceptions.RequestException as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution fractured: {e}")
        if e.response is not None:
            print("Underlying Exception Traces:")
            print(e.response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Setu AA live ingestion orchestrator.")
    parser.add_argument("--mobile", help="10-digit mobile number to receive OTP (or set SETU_MOBILE).")
    parser.add_argument("--days", type=int, default=30, help="How many past days to request (default: 30).")
    args = parser.parse_args()
    orchestrate_live_pipeline(mobile=args.mobile, days=args.days)
