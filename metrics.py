"""Metrics collection module."""

from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List


class Metrics:
    """Metrics collector for tracking API usage and performance."""

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = Lock()
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._errors_by_type: Dict[str, int] = defaultdict(int)
        self._response_times: List[float] = []
        self._max_response_times = 1000  # Keep last 1000 response times for average

    def record_request(self, response_time: float, cached: bool = False) -> None:
        """
        Record a successful request.

        Args:
            response_time: Response time in seconds
            cached: Whether the result was from cache
        """
        with self._lock:
            self._total_requests += 1
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            # Store response time (keep only last N)
            self._response_times.append(response_time)
            if len(self._response_times) > self._max_response_times:
                self._response_times.pop(0)

    def record_error(self, error_type: str) -> None:
        """
        Record an error.

        Args:
            error_type: Type of error (timeout, api_error, validation_error, internal_error)
        """
        with self._lock:
            self._total_requests += 1
            self._errors_by_type[error_type] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary with metrics
        """
        with self._lock:
            total_errors = sum(self._errors_by_type.values())
            successful_requests = self._total_requests - total_errors

            # Calculate average response time
            avg_response_time = 0.0
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)

            # Calculate cache hit rate
            cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = 0.0
            if cache_requests > 0:
                cache_hit_rate = self._cache_hits / cache_requests

            return {
                "total_requests": self._total_requests,
                "successful_requests": successful_requests,
                "total_errors": total_errors,
                "errors_by_type": dict(self._errors_by_type),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "average_response_time_seconds": round(avg_response_time, 4),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._errors_by_type.clear()
            self._response_times.clear()


# Global metrics instance
metrics = Metrics()

