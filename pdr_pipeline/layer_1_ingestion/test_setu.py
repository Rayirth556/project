import logging
import json
import requests
from datetime import datetime, timedelta, timezone
from setu_connector import SetuAAConnector

# Setup basic logging to see the output clearly
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_consent_test():
    # 1. Initialize our connector class
    connector = SetuAAConnector()

    # 2. Setup dates dynamically
    now = datetime.now(timezone.utc)
    
    # We want transaction data for the last 1 year up to today
    data_fetch_start = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data_fetch_end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 3. Define the Consent Payload for Setu V2
    payload = {
        "vua": "",
        "consentDuration": {
            "unit": "MONTH",
            "value": 1
        },
        "consentTypes": ["TRANSACTIONS", "PROFILE", "SUMMARY"],
        "fiTypes": ["DEPOSIT"],
        "dataRange": {
            "from": data_fetch_start,
            "to": data_fetch_end
        },
        "purpose": {
            "code": "101",
            "refUri": "https://api.rebit.org.in/aa/purpose/101.xml",
            "text": "Wealth management service",
            "category": {
                "type": "string"
            }
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

    fi_types: list = payload['fiTypes']  # type: ignore[assignment]
    print("\n--- STARTING SETU CONSENT TEST ---")
    print(f"Requesting '{fi_types[0]}' data from: {data_fetch_start} to {data_fetch_end}")

    # 4. Make the request to Setu
    try:
        response = connector.create_consent_request(payload)
        print("\n[SUCCESS] CONSENT CREATED SUCCESSFULLY")
        print("\nResponse from Setu Sandbox:")
        print(json.dumps(response, indent=2))
        
        # Guide the user on the next steps
        consent_url = response.get("url")
        consent_id = response.get("id")
        
        if consent_url:
            print(f"\n\n=======================================================")
            print(f"[*] ACTION REQUIRED:")
            print(f"1. Open this URL in your web browser: \n{consent_url}")
            print(f"2. Follow the Sandbox prompts to 'Approve' or 'Accept' the request.")
            print(f"3. Note down this Consent ID for the next step: {consent_id}")
            print(f"=======================================================\n")
            
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] FAILED TO CREATE CONSENT")
        if e.response is not None:
            print("Response error:", e.response.text)
        print(f"Check your Client ID and Client Secret in setu_connector.py.")

if __name__ == "__main__":
    run_consent_test()
