import json
from pathlib import Path
import os
import uuid

def fetch_financial_info(consent_id: str) -> dict:
    """
    Simulates fetching financial information (FI data) from Finvu after a consent is ACTIVE.
    It reads from the local mock_data.json file.
    """
    # Simulate an error if no consent ID is provided
    if not consent_id:
        return {"status": "ERROR", "message": "Invalid consent ID provided."}
    
    # Path to the mock JSON data
    mock_file_path = Path(__file__).parent / "mock_data.json"
    
    try:
        with open(mock_file_path, 'r', encoding='utf-8') as f:
            mock_data = json.load(f)
            
        return {
            "status": "SUCCESS",
            "session_id": str(uuid.uuid4()),
            "data": mock_data
        }
    except FileNotFoundError:
        return {
            "status": "ERROR", 
            "message": f"Mock data file not found at {mock_file_path}. Please ensure mock_data.json exists."
        }
    except json.JSONDecodeError:
         return {
            "status": "ERROR", 
            "message": "Failed to parse mock_data.json. Invalid JSON format."
        }

if __name__ == "__main__":
    result = fetch_financial_info("sample-consent-id")
    print(f"Fetch Status: {result.get('status')}")
