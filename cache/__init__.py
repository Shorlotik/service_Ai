"""Cache package for different caching strategies."""

from typing import Optional

from config import settings, CacheStrategy as CacheStrategyEnum

from .base import Cache
from .memory import InMemoryCache
from .redis_cache import RedisCache


def create_cache(config: Optional[object] = None) -> Cache:
    """
    Factory function to create cache instance based on configuration.

    Args:
        config: Optional settings object (defaults to global settings)

    Returns:
        Instance of cache (InMemoryCache or RedisCache)

    Raises:
        ValueError: If cache strategy is not supported
    """
    if config is None:
        config = settings

    strategy = config.cache_strategy
    default_ttl = config.cache_ttl

    if strategy == CacheStrategyEnum.MEMORY:
        return InMemoryCache(
            max_size=128,  # Default max size for in-memory cache
            default_ttl=default_ttl,
        )

    elif strategy == CacheStrategyEnum.REDIS:
        return RedisCache(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            default_ttl=default_ttl,
        )

    else:
        raise ValueError(f"Unsupported cache strategy: {strategy}")


__all__ = [
    "Cache",
    "InMemoryCache",
    "RedisCache",
    "create_cache",
]
