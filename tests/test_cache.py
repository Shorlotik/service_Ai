"""Unit tests for cache implementations."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cache.memory import InMemoryCache
from cache.redis_cache import RedisCache


@pytest.mark.asyncio
async def test_in_memory_cache_set_get():
    """Test in-memory cache set and get operations."""
    cache = InMemoryCache(max_size=10, default_ttl=None)

    await cache.set("key1", "value1")
    result = await cache.get("key1")

    assert result == "value1"


@pytest.mark.asyncio
async def test_in_memory_cache_exists():
    """Test in-memory cache exists operation."""
    cache = InMemoryCache(max_size=10, default_ttl=None)

    assert await cache.exists("key1") is False

    await cache.set("key1", "value1")
    assert await cache.exists("key1") is True


@pytest.mark.asyncio
async def test_in_memory_cache_delete():
    """Test in-memory cache delete operation."""
    cache = InMemoryCache(max_size=10, default_ttl=None)

    await cache.set("key1", "value1")
    assert await cache.exists("key1") is True

    await cache.delete("key1")
    assert await cache.exists("key1") is False
    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiration():
    """Test in-memory cache TTL expiration."""
    cache = InMemoryCache(max_size=10, default_ttl=1)  # 1 second TTL

    await cache.set("key1", "value1", ttl=1)
    assert await cache.get("key1") == "value1"

    # Wait for expiration
    time.sleep(1.1)

    assert await cache.get("key1") is None
    assert await cache.exists("key1") is False


@pytest.mark.asyncio
async def test_in_memory_cache_lru_eviction():
    """Test in-memory cache LRU eviction."""
    cache = InMemoryCache(max_size=2, default_ttl=None)

    # Fill cache to max size
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    # Access key1 to make it more recently used
    await cache.get("key1")

    # Add new key - should evict key2 (least recently used)
    await cache.set("key3", "value3")

    assert await cache.get("key1") == "value1"  # Should still exist
    assert await cache.get("key2") is None  # Should be evicted
    assert await cache.get("key3") == "value3"  # Should exist


@pytest.mark.asyncio
async def test_in_memory_cache_clear():
    """Test in-memory cache clear operation."""
    cache = InMemoryCache(max_size=10, default_ttl=None)

    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    await cache.clear()

    assert await cache.get("key1") is None
    assert await cache.get("key2") is None
    assert await cache.exists("key1") is False


@pytest.mark.asyncio
async def test_redis_cache_set_get():
    """Test Redis cache set and get operations."""
    cache = RedisCache(host="localhost", port=6379, db=0, default_ttl=None)

    # Mock Redis client
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=b'{"test": "value"}')
    mock_client.set = AsyncMock()
    mock_client.setex = AsyncMock()
    mock_client.exists = AsyncMock(return_value=1)
    mock_client.delete = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("cache.redis_cache.redis.from_url", return_value=mock_client):
        # Mock _get_client to return our mock
        cache._get_client = AsyncMock(return_value=mock_client)

        await cache.set("key1", {"test": "value"})
        result = await cache.get("key1")

        assert result == {"test": "value"}
        mock_client.set.assert_called_once()


@pytest.mark.asyncio
async def test_redis_cache_exists():
    """Test Redis cache exists operation."""
    cache = RedisCache(host="localhost", port=6379, db=0)

    mock_client = AsyncMock()
    mock_client.exists = AsyncMock(return_value=1)
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("cache.redis_cache.redis.from_url", return_value=mock_client):
        cache._get_client = AsyncMock(return_value=mock_client)

        result = await cache.exists("key1")

        assert result is True
        mock_client.exists.assert_called_once_with("key1")


@pytest.mark.asyncio
async def test_redis_cache_delete():
    """Test Redis cache delete operation."""
    cache = RedisCache(host="localhost", port=6379, db=0)

    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("cache.redis_cache.redis.from_url", return_value=mock_client):
        cache._get_client = AsyncMock(return_value=mock_client)

        await cache.delete("key1")

        mock_client.delete.assert_called_once_with("key1")


@pytest.mark.asyncio
async def test_redis_cache_ttl():
    """Test Redis cache TTL support."""
    cache = RedisCache(host="localhost", port=6379, db=0, default_ttl=60)

    mock_client = AsyncMock()
    mock_client.setex = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("cache.redis_cache.redis.from_url", return_value=mock_client):
        cache._get_client = AsyncMock(return_value=mock_client)

        await cache.set("key1", "value1", ttl=120)

        mock_client.setex.assert_called_once()
        # Check that setex was called with correct TTL
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 120  # TTL value


@pytest.mark.asyncio
async def test_redis_cache_ping():
    """Test Redis cache ping operation."""
    cache = RedisCache(host="localhost", port=6379, db=0)

    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()

    with patch("cache.redis_cache.redis.from_url", return_value=mock_client):
        cache._get_client = AsyncMock(return_value=mock_client)

        result = await cache.ping()

        assert result is True
        mock_client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_redis_cache_connection_error():
    """Test Redis cache connection error handling."""
    cache = RedisCache(host="localhost", port=6379, db=0)

    with patch("cache.redis_cache.redis.from_url", side_effect=Exception("Connection failed")):
        cache._get_client = AsyncMock(side_effect=Exception("Connection failed"))

        with pytest.raises(Exception) as exc_info:
            await cache.get("key1")

        assert "Connection failed" in str(exc_info.value) or "Failed to connect to Redis" in str(exc_info.value)

