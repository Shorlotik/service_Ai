"""Hugging Face Inference API provider."""

import httpx
from typing import Dict, Any

from .base import MLProvider


class HuggingFaceProvider(MLProvider):
    """Provider for Hugging Face Inference API."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_url: str = "https://api-inference.huggingface.co/models",
        timeout: int = 30,
    ):
        """
        Initialize Hugging Face provider.

        Args:
            api_key: Hugging Face API key
            model_name: Name of the model (e.g., "cardiffnlp/twitter-roberta-base-sentiment-latest")
            api_url: Base URL for Hugging Face API
            timeout: Request timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.endpoint = f"{self.api_url}/{self.model_name}"

    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using Hugging Face Inference API.

        Args:
            text: Text to classify

        Returns:
            Dictionary with raw response from Hugging Face API

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPStatusError: If API returns error status
            Exception: For other errors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"inputs": text}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # Hugging Face API may return 503 if model is loading
                if e.response.status_code == 503:
                    error_msg = "Model is currently loading, please try again later"
                    raise Exception(error_msg) from e
                # Handle other HTTP errors
                error_text = e.response.text if e.response else "Unknown error"
                raise Exception(
                    f"Hugging Face API error ({e.response.status_code}): {error_text}"
                ) from e
            except httpx.RequestError as e:
                raise Exception(f"Failed to connect to Hugging Face API: {str(e)}") from e

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "huggingface"

