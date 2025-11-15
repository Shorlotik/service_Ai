"""In-memory cache implementation."""

import time
from typing import Any, Optional, Dict, Tuple

from .base import Cache


class InMemoryCache(Cache):
    """In-memory cache implementation with TTL support."""

    def __init__(self, max_size: int = 128, default_ttl: Optional[int] = None):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of cached items (for LRU eviction)
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self._cache: Dict[str, Tuple[Any, Optional[float]]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._access_times: Dict[str, float] = {}  # For LRU tracking

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key not in self._cache:
            return None

        value, expiry_time = self._cache[key]

        # Check if expired
        if expiry_time is not None and time.time() > expiry_time:
            # Remove expired entry
            await self.delete(key)
            return None

        # Update access time for LRU
        self._access_times[key] = time.time()

        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default_ttl if None)
        """
        # Use provided TTL or default
        cache_ttl = ttl if ttl is not None else self.default_ttl

        # Calculate expiry time
        expiry_time = None
        if cache_ttl is not None:
            expiry_time = time.time() + cache_ttl

        # Check if we need to evict (LRU)
        if len(self._cache) >= self.max_size and key not in self._cache:
            await self._evict_lru()

        # Store value with expiry time
        self._cache[key] = (value, expiry_time)
        self._access_times[key] = time.time()

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired, False otherwise
        """
        if key not in self._cache:
            return False

        _, expiry_time = self._cache[key]

        # Check if expired
        if expiry_time is not None and time.time() > expiry_time:
            await self.delete(key)
            return False

        return True

    async def delete(self, key: str) -> None:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete
        """
        self._cache.pop(key, None)
        self._access_times.pop(key, None)

    async def _evict_lru(self) -> None:
        """Evict least recently used item from cache."""
        if not self._access_times:
            # If no access times, remove first item
            if self._cache:
                key = next(iter(self._cache))
                await self.delete(key)
            return

        # Find least recently used key
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        await self.delete(lru_key)

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache (synchronous helper)."""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, expiry_time) in self._cache.items()
            if expiry_time is not None and current_time > expiry_time
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)

