from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

app = FastAPI(
    title="Finvu AA Simulator", 
    description="Simulates the RBI Account Aggregator (AA) architecture for alternative credit scoring"
)

# In-memory storage for simplicity (in a real app, use a database)
consents_db = {}
sessions_db = {}

# --- Pydantic Models for Input Validation ---

class ConsentRequest(BaseModel):
    user_id: str
    fip_id: str = "SIMULATED-FIP"
    
class ConsentApproval(BaseModel):
    consentHandle: str

class FIRequest(BaseModel):
    consentId: str
    # dateRange is commonly structured as {"from": "ISO8601", "to": "ISO8601"}
    dateRange: dict

class FIFetchRequest(BaseModel):
    sessionId: str
    use_static_mock: bool = False # Optional flag to return static mock data

# --- Synthetic Data Generator ---

def generate_synthetic_transactions(days_history: int = 90) -> tuple[float, List[Dict[str, Any]]]:
    """
    Generates realistic synthetic banking transactions.
    Includes salary credits, utility bills, and random daily transactions.
    """
    transactions = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_history)
    
    current_date = start_date
    current_balance = round(random.uniform(20000, 100000), 2) # Initial balance
    
    categories = ["grocery", "shopping", "fuel", "food", "transfer"]
    
    while current_date <= end_date:
        # 1. Salary credit on the 1st of each month
        if current_date.day == 1:
            salary = round(random.uniform(30000, 80000), 2)
            current_balance += salary
            transactions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "amount": salary,
                "type": "credit",
                "category": "salary"
            })
            
        # 2. Utility bill payment around the 5th
        if current_date.day == 5:
            # Assume an 80% chance of paying utility bill on the 5th
            if random.random() < 0.8:
                utility = round(random.uniform(1000, 5000), 2)
                current_balance -= utility
                if current_balance < 0: 
                    current_balance += utility # Avoid negative balance for simplicity here
                else:
                    transactions.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "amount": utility,
                        "type": "debit",
                        "category": "utilities"
                    })
        
        # 3. Random daily transactions
        # Determine number of transactions today (0 to 3)
        num_daily_txns = random.randint(0, 3)
        for _ in range(num_daily_txns):
            # 20% chance of credit, 80% chance of debit for daily random txn
            if random.random() < 0.2:
                 amount = round(random.uniform(500, 5000), 2)
                 current_balance += amount
                 transactions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "amount": amount,
                    "type": "credit",
                    "category": "transfer" # Receiving money
                 })
            else:
                 amount = round(random.uniform(100, 3000), 2)
                 current_balance -= amount
                 # Simple overdraft protection for the simulation
                 if current_balance < 0:
                     current_balance += amount
                 else:
                     transactions.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "amount": amount,
                        "type": "debit",
                        "category": random.choice(categories)
                     })

        current_date += timedelta(days=1)
        
    # Sort transactions by date (descending, newest first - common in bank statements)
    transactions.sort(key=lambda x: x["date"], reverse=True)
    
    return round(current_balance, 2), transactions

# --- API Endpoints ---

@app.post("/Consent")
async def create_consent(request: ConsentRequest):
    """
    1. POST /Consent
    Generates a unique consentHandle (UUID) and returns status "PENDING".
    """
    consent_handle = str(uuid.uuid4())
    consents_db[consent_handle] = {
        "user_id": request.user_id,
        "fip_id": request.fip_id,
        "status": "PENDING"
    }
    return {
        "ConsentHandle": consent_handle,
        "status": "PENDING"
    }

@app.post("/Consent/handle")
async def approve_consent(request: ConsentApproval):
    """
    2. POST /Consent/handle
    Simulates user approval of consent. Returns a consentId and status "ACTIVE".
    """
    handle = request.consentHandle
    if handle not in consents_db:
        raise HTTPException(status_code=404, detail="Consent handle not found")
        
    consent_data = consents_db[handle]
    if consent_data["status"] != "PENDING":
        raise HTTPException(status_code=400, detail="Consent is not in PENDING state")
        
    consent_id = str(uuid.uuid4())
    # Update state: The handle is now resolved to an ACTIVE Consent ID
    consents_db[consent_id] = {
        "handle": handle,
        "status": "ACTIVE",
        "user_id": consent_data["user_id"]
    }
    
    return {
        "consentId": consent_id,
        "status": "ACTIVE"
    }

@app.post("/FI/request")
async def request_financial_info(request: FIRequest):
    """
    3. POST /FI/request
    Simulates the FIU requesting financial information from the Account Aggregator.
    Returns a sessionId and status "DATA_READY".
    """
    consent_id = request.consentId
    if consent_id not in consents_db or consents_db[consent_id]["status"] != "ACTIVE":
         raise HTTPException(status_code=403, detail="Invalid or inactive consent ID")
         
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "consentId": consent_id,
        "status": "DATA_READY"
    }
    
    return {
        "sessionId": session_id,
        "status": "DATA_READY"
    }

@app.post("/FI/fetch")
async def fetch_financial_info(request: FIFetchRequest):
    """
    4. POST /FI/fetch
    Returns financial data for a user account.
    If 'use_static_mock' is True, reads from local mock_data.json. 
    Otherwise, generates dynamic synthetic data.
    """
    session_id = request.sessionId
    if session_id not in sessions_db or sessions_db[session_id]["status"] != "DATA_READY":
         raise HTTPException(status_code=403, detail="Invalid session ID or data not ready")
         
    # Fallback/Optional feature: return static mock file
    if request.use_static_mock:
        import json
        from pathlib import Path
        mock_path = Path(__file__).parent / "mock_data.json"
        try:
            with open(mock_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load static mock data: {e}")
         
    # Generate dynamic, realistic synthetic data on the fly (90 days history)
    final_balance, transactions = generate_synthetic_transactions(days_history=90)
    
    response_data = {
        "accounts": [
            {
                "type": "savings",
                "balance": final_balance,
                "transactions": transactions
            }
        ]
    }
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    # Typically run with: uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
