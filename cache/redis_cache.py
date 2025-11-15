"""Redis cache implementation."""

import json
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from .base import Cache


class RedisCache(Cache):
    """Redis cache implementation with TTL support."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self._client: Optional[Redis] = None

    async def _get_client(self) -> Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                decode_responses=False,  # We'll handle encoding ourselves
            )
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            client = await self._get_client()
            value = await client.get(key)
            if value is None:
                return None

            # Deserialize JSON value
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # If not JSON, return as string
                return value.decode("utf-8") if isinstance(value, bytes) else value

        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Redis get error: {str(e)}") from e

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (uses default_ttl if None)
        """
        try:
            client = await self._get_client()

            # Serialize value to JSON
            if isinstance(value, (str, int, float, bool, type(None))):
                # Simple types can be stored directly
                serialized_value = json.dumps(value)
            else:
                # Complex types need JSON serialization
                serialized_value = json.dumps(value)

            # Use provided TTL or default
            cache_ttl = ttl if ttl is not None else self.default_ttl

            if cache_ttl is not None:
                await client.setex(key, cache_ttl, serialized_value)
            else:
                await client.set(key, serialized_value)

        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise Exception(f"Failed to serialize value for cache: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Redis set error: {str(e)}") from e

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            client = await self._get_client()
            result = await client.exists(key)
            return bool(result)

        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Redis exists error: {str(e)}") from e

    async def delete(self, key: str) -> None:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete
        """
        try:
            client = await self._get_client()
            await client.delete(key)

        except redis.ConnectionError as e:
            raise Exception(f"Failed to connect to Redis: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Redis delete error: {str(e)}") from e

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def ping(self) -> bool:
        """
        Check if Redis connection is alive.

        Returns:
            True if connection is alive, False otherwise
        """
        try:
            client = await self._get_client()
            result = await client.ping()
            return bool(result)
        except Exception:
            return False

