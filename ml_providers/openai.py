"""OpenAI API provider for text classification."""

import httpx
import json
from typing import Dict, Any

from .base import MLProvider


class OpenAIProvider(MLProvider):
    """Provider for OpenAI API (using chat completions for classification)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        api_url: str = "https://api.openai.com/v1/chat/completions",
        timeout: int = 30,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: OpenAI model name (default: gpt-3.5-turbo)
            api_url: OpenAI API endpoint URL
            timeout: Request timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using OpenAI API.

        Uses chat completions API with a prompt for sentiment analysis.
        Returns classification in JSON format.

        Args:
            text: Text to classify

        Returns:
            Dictionary with classification result

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPStatusError: If API returns error status
            Exception: For other errors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prompt for sentiment classification
        system_prompt = (
            "You are a sentiment analysis classifier. "
            "Analyze the given text and classify it as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Respond with a JSON object containing 'label' (one of: POSITIVE, NEGATIVE, NEUTRAL) "
            "and 'confidence' (a float between 0.0 and 1.0)."
        )

        user_prompt = f"Classify the following text: {text}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Lower temperature for more consistent classification
            "response_format": {"type": "json_object"},  # Request JSON response
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Extract content from OpenAI response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if content:
                        try:
                            # Parse JSON response from OpenAI
                            classification = json.loads(content)
                            return classification
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try to extract label from text
                            return self._parse_text_response(content)
                    else:
                        raise Exception("Empty response from OpenAI API")
                else:
                    raise Exception("Invalid response format from OpenAI API")

            except httpx.HTTPStatusError as e:
                error_data = {}
                try:
                    error_data = e.response.json() if e.response else {}
                except Exception:
                    pass

                error_message = error_data.get("error", {}).get("message", "Unknown error")
                raise Exception(
                    f"OpenAI API error ({e.response.status_code}): {error_message}"
                ) from e
            except httpx.RequestError as e:
                raise Exception(f"Failed to connect to OpenAI API: {str(e)}") from e

    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """
        Parse text response if JSON parsing failed.

        Tries to extract label and confidence from text response.

        Args:
            content: Text content from OpenAI

        Returns:
            Dictionary with label and confidence
        """
        content_upper = content.upper()
        label = "NEUTRAL"
        confidence = 0.5

        # Try to find label
        if "POSITIVE" in content_upper:
            label = "POSITIVE"
            confidence = 0.9
        elif "NEGATIVE" in content_upper:
            label = "NEGATIVE"
            confidence = 0.9

        # Try to extract confidence number
        import re

        confidence_match = re.search(r"confidence[:\s]+([0-9.]+)", content_upper)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0  # Convert percentage to decimal
            except ValueError:
                pass

        return {"label": label, "confidence": confidence}

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"

