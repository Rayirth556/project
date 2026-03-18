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

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _raise_for_status_with_body(self, resp: requests.Response) -> None:
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = resp.text
            except Exception:
                body = "<unable to read response body>"
            logger.error(f"HTTP {resp.status_code} for {resp.request.method} {resp.url}\nResponse body: {body}")
            raise e

    def _request_with_fallback(self, method: str, endpoints: list[str], *, json: dict | None = None) -> dict:
        """
        Try multiple endpoints in order, returning the first successful JSON response.
        Useful because Setu exposes different paths across environments/products.
        """
        last_exc: Exception | None = None
        for endpoint in endpoints:
            try:
                resp = requests.request(method, endpoint, headers=self.headers, json=json)
                self._raise_for_status_with_body(resp)
                return resp.json()
            except requests.exceptions.HTTPError as e:
                last_exc = e
                status = getattr(resp, "status_code", None)

                # Only fall back when it looks like the endpoint/path is unsupported
                # (or when credentials are scoped differently across variants).
                if status in (404, 405):
                    continue
                if status in (401, 403):
                    # Try the next endpoint variant if available.
                    continue

                # For "real" request errors (400, 409, 5xx, etc), don't mask the primary error
                # by falling back to a different endpoint.
                raise
            except requests.exceptions.RequestException as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No endpoints provided")

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
            self._raise_for_status_with_body(response)
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
            self._raise_for_status_with_body(response)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch consent status for '{consent_id}': {e}")
            raise

    def create_data_session(self, consent_id: str, data_range: dict) -> dict:
        """
        Creates a data session for an ACTIVE consent, enabling the fetching of actual FI data.
        """
        logger.info(f"Creating data session for consent_id: {consent_id}")
        
        payload = {
            "consentId": consent_id,
            "dataRange": data_range,
            "format": "json"
        }
        
        try:
            # Prefer v2 sessions (commonly used with Bridge-style credentials),
            # fall back to legacy/non-v2 if the environment supports it.
            return self._request_with_fallback(
                "POST",
                [
                    f"{self.base_url}/v2/sessions",
                    f"{self.base_url}/sessions",
                ],
                json=payload,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create data session: {e}")
            raise

    def get_session_status(self, session_id: str) -> dict:
        try:
            return self._request_with_fallback(
                "GET",
                [
                    f"{self.base_url}/v2/sessions/{session_id}",
                    f"{self.base_url}/sessions/{session_id}",
                ],
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check session status: {e}")
            raise

    def fetch_fi_data(self, session_id: str) -> dict:
        """
        Fetches decrypted FI data for a data session.
        Setu returns FI data on the same endpoint used for session status.
        """
        logger.info(f"Fetching FI data for session_id: {session_id}")

        try:
            data = self._request_with_fallback(
                "GET",
                [
                    f"{self.base_url}/v2/sessions/{session_id}",
                    f"{self.base_url}/sessions/{session_id}",
                ],
            )
            logger.info(f"Successfully retrieved FI data for session_id: {session_id}.")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch FI data for session_id '{session_id}': {e}")
            raise
