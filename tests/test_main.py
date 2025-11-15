"""Unit tests for main FastAPI application."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from main import app
from models import ClassifyResponseSimple


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_ml_provider():
    """Create mock ML provider."""
    provider = AsyncMock()
    provider.get_provider_name.return_value = "test_provider"
    provider.classify = AsyncMock(
        return_value=[
            {"label": "POSITIVE", "score": 0.95},
            {"label": "NEGATIVE", "score": 0.05},
        ]
    )
    return provider


@pytest.fixture
def mock_cache():
    """Create mock cache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # Cache miss by default
    cache.set = AsyncMock()
    cache.exists = AsyncMock(return_value=False)
    cache.ping = AsyncMock(return_value=True)
    return cache


@pytest.mark.asyncio
async def test_classify_success(client, mock_ml_provider, mock_cache):
    """Test successful classification request."""
    # Mock the global instances
    with patch("main.ml_provider", mock_ml_provider), patch(
        "main.cache", mock_cache
    ), patch(
        "ml_providers.normalizers.normalize_response",
        return_value=("POSITIVE", 0.95, [("POSITIVE", 0.95), ("NEGATIVE", 0.05)]),
    ):
        response = client.post(
            "/classify",
            json={"text": "I love this product!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert "cached" in data
        assert data["label"] == "POSITIVE"
        assert data["confidence"] == 0.95
        assert data["cached"] is False

        # Verify ML provider was called
        mock_ml_provider.classify.assert_called_once_with("I love this product!")

        # Verify result was cached
        mock_cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_classify_cache_hit(client, mock_ml_provider, mock_cache):
    """Test classification with cache hit."""
    # Setup cache to return cached result
    cached_result = {
        "label": "POSITIVE",
        "confidence": 0.95,
        "labels": [{"label": "POSITIVE", "confidence": 0.95}],
    }
    mock_cache.get = AsyncMock(return_value=cached_result)

    with patch("main.ml_provider", mock_ml_provider), patch("main.cache", mock_cache):
        response = client.post(
            "/classify",
            json={"text": "I love this product!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert data["label"] == "POSITIVE"
        assert data["confidence"] == 0.95

        # Verify ML provider was NOT called (cache hit)
        mock_ml_provider.classify.assert_not_called()

        # Verify cache was checked
        mock_cache.get.assert_called_once()


@pytest.mark.asyncio
async def test_classify_validation_error(client):
    """Test classification with validation error."""
    # Empty text should fail validation
    response = client.post(
        "/classify",
        json={"text": ""},
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"] == "validation_error"


@pytest.mark.asyncio
async def test_health_check(client, mock_cache):
    """Test health check endpoint."""
    with patch("main.cache", mock_cache), patch(
        "main.ml_provider", MagicMock(get_provider_name=lambda: "test")
    ):
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.text)
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "cache_hits" in data
    assert "cache_misses" in data
    assert "average_response_time_seconds" in data


@pytest.mark.asyncio
async def test_classify_timeout_error(client, mock_ml_provider, mock_cache):
    """Test classification with timeout error."""
    # Mock timeout exception
    mock_ml_provider.classify = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

    with patch("main.ml_provider", mock_ml_provider), patch("main.cache", mock_cache):
        response = client.post(
            "/classify",
            json={"text": "Test text"},
        )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"] == "timeout"
        assert "message" in data
        assert "Request to ML API timed out" in data["message"]


@pytest.mark.asyncio
async def test_classify_api_error(client, mock_ml_provider, mock_cache):
    """Test classification with API error (HTTP status error)."""
    # Mock HTTP status error
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_ml_provider.classify = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
    )

    with patch("main.ml_provider", mock_ml_provider), patch("main.cache", mock_cache):
        response = client.post(
            "/classify",
            json={"text": "Test text"},
        )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"] == "api_error"
        assert "message" in data
        assert "ML API returned error status" in data["message"]


@pytest.mark.asyncio
async def test_classify_network_error(client, mock_ml_provider, mock_cache):
    """Test classification with network/connection error."""
    # Mock connection error
    mock_ml_provider.classify = AsyncMock(
        side_effect=httpx.ConnectError("Connection failed")
    )

    with patch("main.ml_provider", mock_ml_provider), patch("main.cache", mock_cache):
        response = client.post(
            "/classify",
            json={"text": "Test text"},
        )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"] == "api_error"
        assert "message" in data
        assert "Failed to connect to ML API" in data["message"]

