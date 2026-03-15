import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_test():
    print("====================================")
    print("FINVU AA API SIMULATOR END-TO-END TEST")
    print("====================================\n")

    # Step 1: POST /Consent
    print("Step 1: Simulating Consent Generation (POST /Consent)")
    payload_consent = {
        "user_id": "test_user_789",
        "fip_id": "HACKATHON_FIP"
    }
    r1 = requests.post(f"{BASE_URL}/Consent", json=payload_consent)
    r1.raise_for_status()
    res1 = r1.json()
    print(f"Response:\n{json.dumps(res1, indent=2)}\n")
    consent_handle = res1["ConsentHandle"]

    time.sleep(1)

    # Step 2: POST /Consent/handle
    print("Step 2: Simulating User Approval (POST /Consent/handle)")
    payload_approval = {
        "consentHandle": consent_handle
    }
    r2 = requests.post(f"{BASE_URL}/Consent/handle", json=payload_approval)
    r2.raise_for_status()
    res2 = r2.json()
    print(f"Response:\n{json.dumps(res2, indent=2)}\n")
    consent_id = res2["consentId"]
    
    time.sleep(1)

    # Step 3: POST /FI/request
    print("Step 3: Requesting FI Data (POST /FI/request)")
    payload_fi_req = {
        "consentId": consent_id,
        "dateRange": {
            "from": "2025-01-01T00:00:00Z",
            "to": "2026-03-15T00:00:00Z"
        }
    }
    r3 = requests.post(f"{BASE_URL}/FI/request", json=payload_fi_req)
    r3.raise_for_status()
    res3 = r3.json()
    print(f"Response:\n{json.dumps(res3, indent=2)}\n")
    session_id = res3["sessionId"]
    
    time.sleep(1)

    # Step 4: POST /FI/fetch
    print("Step 4: Fetching FI Data (POST /FI/fetch)")
    payload_fetch = {
        "sessionId": session_id
    }
    
    # We optionally allow passing 'use_static_mock' boolean if the backend supports it
    # payload_fetch["use_static_mock"] = True 
    
    r4 = requests.post(f"{BASE_URL}/FI/fetch", json=payload_fetch)
    r4.raise_for_status()
    res4 = r4.json()
    
    # Just print the first few transactions so the console isn't flooded
    print("Response Status: 200 OK")
    account = res4.get("accounts", [{}])[0]
    txns = account.get("transactions", [])
    
    print(f"Account Type: {account.get('type')}")
    print(f"Final Balance: ${account.get('balance')}")
    print(f"Total Transactions Generated: {len(txns)}")
    
    print("\nPreview of first 5 transactions:")
    print(json.dumps(txns[:5], indent=2))
    
    print("\nTest Complete! AA Architecture Simulation is Functional!")

if __name__ == "__main__":
    try:
        run_test()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the FastAPI server.")
        print("Make sure you have started the server using: uvicorn main:app --reload")
