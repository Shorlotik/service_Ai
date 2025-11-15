"""Local ML model provider (for models running in Docker containers)."""

import httpx
from typing import Dict, Any

from .base import MLProvider


class LocalMLProvider(MLProvider):
    """Provider for local ML models accessible via HTTP endpoint."""

    def __init__(self, model_url: str, timeout: int = 30):
        """
        Initialize local ML provider.

        Args:
            model_url: URL of the local ML model endpoint
            timeout: Request timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.model_url = model_url.rstrip("/")

    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using local ML model.

        Args:
            text: Text to classify

        Returns:
            Dictionary with raw response from local ML model

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPStatusError: If API returns error status
            Exception: For other errors
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Common formats for local model endpoints
            # Try POST with JSON body first
            try:
                response = await client.post(
                    self.model_url,
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # If 405 Method Not Allowed, try GET with query parameter
                if e.response.status_code == 405:
                    response = await client.get(
                        self.model_url,
                        params={"text": text},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    return response.json()
                raise
            except httpx.RequestError as e:
                raise Exception(f"Failed to connect to local model: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "local"

