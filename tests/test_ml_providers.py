"""Unit tests for ML providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ml_providers.local import LocalMLProvider
from ml_providers.huggingface import HuggingFaceProvider
from ml_providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_local_provider_success():
    """Test local ML provider with successful response."""
    provider = LocalMLProvider(model_url="http://localhost:8000/predict", timeout=30)

    mock_response = MagicMock()
    mock_response.json.return_value = {"label": "POSITIVE", "confidence": 0.95}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await provider.classify("Test text")

        assert result == {"label": "POSITIVE", "confidence": 0.95}
        mock_client.post.assert_called_once()
        assert mock_client.post.call_args[1]["json"] == {"text": "Test text"}


@pytest.mark.asyncio
async def test_local_provider_timeout():
    """Test local ML provider with timeout."""
    provider = LocalMLProvider(model_url="http://localhost:8000/predict", timeout=5)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

        # Provider wraps TimeoutException in Exception
        with pytest.raises(Exception) as exc_info:
            await provider.classify("Test text")
        
        assert "Failed to connect to local model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_huggingface_provider_success():
    """Test Hugging Face provider with successful response."""
    provider = HuggingFaceProvider(
        api_key="test_key",
        model_name="test/model",
        timeout=30,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"label": "POSITIVE", "score": 0.95},
        {"label": "NEGATIVE", "score": 0.05},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await provider.classify("Test text")

        assert isinstance(result, list)
        assert len(result) == 2
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {"inputs": "Test text"}
        assert "Authorization" in call_args[1]["headers"]


@pytest.mark.asyncio
async def test_huggingface_provider_503_error():
    """Test Hugging Face provider with 503 error (model loading)."""
    provider = HuggingFaceProvider(
        api_key="test_key",
        model_name="test/model",
        timeout=30,
    )

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Model is loading"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "503", request=MagicMock(), response=mock_response
            )
        )

        with pytest.raises(Exception) as exc_info:
            await provider.classify("Test text")

        assert "Model is currently loading" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_provider_success():
    """Test OpenAI provider with successful response."""
    provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo", timeout=30)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"label": "POSITIVE", "confidence": 0.95}'
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await provider.classify("Test text")

        assert isinstance(result, dict)
        assert "label" in result or "choices" in result
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "Authorization" in call_args[1]["headers"]


@pytest.mark.asyncio
async def test_openai_provider_api_error():
    """Test OpenAI provider with API error."""
    provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo", timeout=30)

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response
            )
        )

        with pytest.raises(Exception) as exc_info:
            await provider.classify("Test text")

        assert "OpenAI API error" in str(exc_info.value)


def test_provider_names():
    """Test provider name methods."""
    local_provider = LocalMLProvider(model_url="http://localhost:8000")
    assert local_provider.get_provider_name() == "local"

    hf_provider = HuggingFaceProvider(api_key="test", model_name="test/model")
    assert hf_provider.get_provider_name() == "huggingface"

    openai_provider = OpenAIProvider(api_key="test")
    assert openai_provider.get_provider_name() == "openai"

