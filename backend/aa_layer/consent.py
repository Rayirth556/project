import uuid
from datetime import datetime, timedelta

def create_consent_request(user_id: str, fip_id: str = "FINVU-FIP") -> dict:
    """
    Simulates sending a consent request to Finvu AA.
    Returns a mock consent payload with a URL and generated ID.
    """
    consent_id = str(uuid.uuid4())
    # Generate a mock expiration date 24 hours from now
    expiry = (datetime.utcnow() + timedelta(days=1)).isoformat() + "Z"

    mock_response = {
        "status": "SUCCESS",
        "consentId": consent_id,
        "consentUrl": f"https://sandbox.finvu.in/consent/{consent_id}",
        "statusCheckInterval": 5, # seconds
        "details": {
            "userId": user_id,
            "fipId": fip_id,
            "expiresAt": expiry,
            "dataTimeRange": {
                "from": (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z", # Past year data
                "to": datetime.utcnow().isoformat() + "Z"
            }
        }
    }
    
    return mock_response

def check_consent_status(consent_id: str) -> dict:
    """
    Simulates polling Finvu for consent status.
    For this mock, we'll arbitrarily say it's ACTIVE, simulating user approval.
    """
    return {
        "consentId": consent_id,
        "status": "ACTIVE",
        "message": "User has approved the consent request."
    }

if __name__ == "__main__":
    print(create_consent_request("user123@finvu"))
