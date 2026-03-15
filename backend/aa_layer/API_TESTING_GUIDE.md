# Testing the Finvu AA API Simulator

This guide will help you test the Fintech Hackathon simulated Account Aggregator API using Python FastAPI.

## Prerequisites

1.  Make sure you are in the correct directory:
    ```bash
    cd backend/aa_layer
    ```
2.  Install the requirements if you haven't already:
    ```bash
    pip install -r requirements.txt
    ```

## 1. Running the FastAPI Backend

Start the development server using uvicorn. Note: Depending on your Python installation, you might need to use `python -m uvicorn` instead of just `uvicorn`.

```bash
uvicorn main:app --reload
```
You should see output like: `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

## 2. Using the Swagger Testing Interface (Recommended)

FastAPI automatically generates an interactive documentation page.

1.  Open your browser and navigate to: **http://127.0.0.1:8000/docs**
2.  You will see the interactive Swagger UI.
3.  Expand each endpoint, click **"Try it out"**, edit the Request body if necessary, and click **"Execute"**.

**Test the endpoints strictly in this order:**

### Step 1: POST /Consent
*   **Action**: Click "Try it out" -> "Execute"
*   **Expected Response**:
    ```json
    {
      "ConsentHandle": "some-uuid-value",
      "status": "PENDING"
    }
    ```
*   **Save the `ConsentHandle` string generated here.**

### Step 2: POST /Consent/handle
*   **Action**: Click "Try it out". In the request body, replace `"string"` with the `ConsentHandle` from Step 1. -> "Execute"
    ```json
    {
      "consentHandle": "the-uuid-from-step-1"
    }
    ```
*   **Expected Response**:
    ```json
    {
      "consentId": "CONSENT_001_or_uuid",
      "status": "ACTIVE"
    }
    ```
*   **Save the `consentId` string generated here.**

### Step 3: POST /FI/request
*   **Action**: Click "Try it out". Focus on the `consentId` field.
    ```json
    {
      "consentId": "the-uuid-from-step-2",
      "dateRange": {
          "from": "2025-01-01",
          "to": "2026-03-15"
      }
    }
    ```
*   **Expected Response**:
    ```json
    {
      "sessionId": "SESSION_123_or_uuid",
      "status": "DATA_READY"
    }
    ```
*   **Save the `sessionId` string generated here.**

### Step 4: POST /FI/fetch
*   **Action**: Click "Try it out". Provide the session ID.
    ```json
    {
      "sessionId": "the-uuid-from-step-3"
    }
    ```
*   **Expected Response**: You should see a large JSON array of synthetically generated financial transactions under `accounts[0].transactions`.

## 3. Quick Terminal Test using cURL

If you prefer testing via terminal commands without the UI, run these one by one (replace the placeholders with the actual UUIDs returned by the previous command).

1.  **Consent Generation:**
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/Consent' -H 'Content-Type: application/json' -d '{"user_id": "user123", "fip_id": "SIMULATED-FIP"}'
    ```

2.  **Consent Approval (replace `YOUR_CONSENT_HANDLE`):**
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/Consent/handle' -H 'Content-Type: application/json' -d '{"consentHandle": "YOUR_CONSENT_HANDLE"}'
    ```

3.  **Data Request (replace `YOUR_CONSENT_ID`):**
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/FI/request' -H 'Content-Type: application/json' -d '{"consentId": "YOUR_CONSENT_ID", "dateRange": {"from": "2025-01-01", "to": "2026-03-15"}}'
    ```

4.  **Data Fetch (replace `YOUR_SESSION_ID`):**
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/FI/fetch' -H 'Content-Type: application/json' -d '{"sessionId": "YOUR_SESSION_ID"}'
    ```

## 4. Helper Script (Python)

If you have Python installed, you can simply run this automated helper script which executes the full flow end-to-end automatically.

Run this script: `python test_aa_api.py`

*(The code for `test_aa_api.py` is included in the project directory).*
