import logging
import requests

logger = logging.getLogger(__name__)


class SetuAAConnector:
    """
    Handles API calls to Setu Account Aggregator Sandbox for consent and data retrieval.
    """
    def __init__(self, base_url: str = "https://fiu-sandbox.setu.co"):
        self.base_url = base_url.rstrip("/")
        # ---------------------------------------------------------
        # ADD YOUR API KEYS HERE:
        # Provide the x-client-id and x-client-secret for Setu AA
        # ---------------------------------------------------------
        self.client_id = "5cd97a89-ad5d-41a6-91b6-07887d7dc6e0"
        self.client_secret = "MVUbfZP217ZSgDhKfNbtw1NuPCTRgq0t"
        self.product_instance_id = "50854c6e-589c-43cb-bdb7-a276cd56086c"
        
        self.headers = {
            "x-client-id": self.client_id,
            "x-client-secret": self.client_secret,
            "x-product-instance-id": self.product_instance_id,
            "Content-Type": "application/json"
        }

    def create_consent_request(self, payload: dict | None = None) -> dict:
        """
        Makes a POST request to the Setu Sandbox to generate a consent link.
        """
        # The payload will contain context for the consent, such as redirect details, 
        # consent artifacts, account types, and time ranges.
        if payload is None:
            payload = {}
            
        endpoint = f"{self.base_url}/v2/consents"
        logger.info(f"Initiating consent request to {endpoint}")
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info("Successfully created consent request.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create consent request: {e}")
            raise

    def get_consent_status(self, consent_id: str) -> dict:
        """
        Polls the Setu API to check the status of a specific consent request (e.g., PENDING, ACTIVE, REJECTED).
        """
        endpoint = f"{self.base_url}/v2/consents/{consent_id}"
        logger.info(f"Checking status for consent_id: {consent_id}")
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch consent status for '{consent_id}': {e}")
            raise

    def create_data_session(self, consent_id: str, data_range: dict) -> dict:
        """
        Creates a data session for an ACTIVE consent, enabling the fetching of actual FI data.
        """
        endpoint = f"{self.base_url}/v2/sessions"
        logger.info(f"Creating data session for consent_id: {consent_id}")
        
        payload = {
            "consentId": consent_id,
            "DataRange": data_range,
            "format": "json"
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create data session: {e}")
            raise

    def fetch_fi_data(self, session_id: str) -> dict:
        """
        Makes a GET request to retrieve the FI data (JSON stream) once consent is approved.
        """
        endpoint = f"{self.base_url}/v2/sessions/{session_id}/fi/data"
        logger.info(f"Fetching FI data for session_id: {session_id}")
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Successfully retrieved FI data for session_id: {session_id}.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch FI data for session_id '{session_id}': {e}")
            raise
