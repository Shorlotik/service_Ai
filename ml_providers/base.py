"""Base abstract class for ML providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class MLProvider(ABC):
    """Abstract base class for ML API providers."""

    def __init__(self, timeout: int = 30):
        """
        Initialize ML provider.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    @abstractmethod
    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using ML API.

        Args:
            text: Text to classify

        Returns:
            Dictionary with raw response from ML API. The format may vary
            depending on the provider, but should contain classification results.

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPStatusError: If API returns error status
            Exception: For other errors
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider name (e.g., "local", "huggingface", "openai")
        """
        pass

