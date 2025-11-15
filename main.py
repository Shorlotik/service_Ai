"""Main FastAPI application for ML API Wrapper Service."""

import json
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError

from config import settings
from logger import logger, setup_logging
from metrics import metrics
from ml_providers import create_provider
from ml_providers.base import MLProvider
from ml_providers.normalizers import normalize_response
from cache import create_cache
from cache.base import Cache
from models import (
    ClassifyRequest,
    ClassifyResponseSimple,
    ClassifyResponseDetailed,
    ErrorResponse,
)
from utils import normalize_text, generate_cache_key


# Global instances (will be initialized in lifespan)
ml_provider: MLProvider = None
cache: Cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown.

    Initializes dependencies on startup and cleans up on shutdown.
    """
    # Startup
    global ml_provider, cache

    logger.info("Starting ML API Wrapper Service...")

    try:
        # Initialize ML provider
        logger.info(f"Initializing ML provider: {settings.ml_provider.value}")
        ml_provider = create_provider()
        logger.info(f"ML provider initialized: {ml_provider.get_provider_name()}")

        # Initialize cache
        logger.info(f"Initializing cache: {settings.cache_strategy.value}")
        cache = create_cache()
        logger.info(f"Cache initialized: {settings.cache_strategy.value}")

        # Test cache connection if Redis
        if settings.cache_strategy.value == "redis":
            try:
                if hasattr(cache, "ping"):
                    is_alive = await cache.ping()
                    if not is_alive:
                        logger.warning("Redis connection test failed, but continuing...")
                    else:
                        logger.info("Redis connection test successful")
            except Exception as e:
                logger.warning(f"Redis connection test error: {e}")

        logger.info("Service started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down ML API Wrapper Service...")

    try:
        # Close cache connection if needed
        if cache and hasattr(cache, "close"):
            await cache.close()
            logger.info("Cache connection closed")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

    logger.info("Service shut down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ML API Wrapper Service for text classification and sentiment analysis",
    lifespan=lifespan,
)

# Setup logging
setup_logging()


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    error_messages = [f"{err['loc']}: {err['msg']}" for err in errors]
    error_message = "; ".join(error_messages)

    logger.warning(f"Validation error: {error_message}")

    metrics.record_error("validation_error")

    return Response(
        content=json.dumps(
            ErrorResponse(
                error="validation_error",
                message="Invalid request data",
                details=error_message,
            ).model_dump()
        ),
        media_type="application/json",
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@app.exception_handler(httpx.TimeoutException)
async def timeout_exception_handler(request: Request, exc: httpx.TimeoutException):
    """Handle timeout errors from ML API."""
    logger.error(f"Timeout error: {exc}", exc_info=True)

    metrics.record_error("timeout")

    return Response(
        content=json.dumps(
            ErrorResponse(
                error="timeout",
                message="Request to ML API timed out",
                details=str(exc),
            ).model_dump()
        ),
        media_type="application/json",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )


@app.exception_handler(httpx.HTTPStatusError)
async def http_status_exception_handler(
    request: Request, exc: httpx.HTTPStatusError
):
    """Handle HTTP errors from ML API."""
    status_code = exc.response.status_code if exc.response else 0
    error_text = exc.response.text if exc.response else "Unknown error"

    logger.error(f"HTTP error from ML API: {status_code} - {error_text}")

    metrics.record_error("api_error")

    return Response(
        content=json.dumps(
            ErrorResponse(
                error="api_error",
                message=f"ML API returned error status {status_code}",
                details=error_text,
            ).model_dump()
        ),
        media_type="application/json",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )


@app.exception_handler(httpx.RequestError)
async def request_exception_handler(request: Request, exc: httpx.RequestError):
    """Handle network/connection errors."""
    logger.error(f"Network error: {exc}", exc_info=True)

    metrics.record_error("api_error")

    return Response(
        content=json.dumps(
            ErrorResponse(
                error="api_error",
                message="Failed to connect to ML API",
                details=str(exc),
            ).model_dump()
        ),
        media_type="application/json",
        status_code=status.HTTP_502_BAD_GATEWAY,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    metrics.record_error("internal_error")

    return Response(
        content=json.dumps(
            ErrorResponse(
                error="internal_error",
                message="An internal server error occurred",
                details=str(exc) if settings.log_level.value == "DEBUG" else None,
            ).model_dump()
        ),
        media_type="application/json",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "ml_provider": settings.ml_provider.value,
        "cache_strategy": settings.cache_strategy.value,
    }


@app.post("/classify", response_model=ClassifyResponseSimple | ClassifyResponseDetailed)
async def classify_text(request: ClassifyRequest):
    """
    Classify text using ML API.

    Args:
        request: Classification request with text

    Returns:
        Classification response with label, confidence, and cached flag
    """
    start_time = time.time()
    cache_key = generate_cache_key(request.text)

    try:
        # Check cache first
        cached_result = await cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for key: {cache_key[:50]}...")
            response_time = time.time() - start_time
            metrics.record_request(response_time, cached=True)

            # Return cached result with cached flag
            if isinstance(cached_result, dict):
                cached_result["cached"] = True
                # Return appropriate format based on settings
                if settings.response_format == "detailed":
                    return ClassifyResponseDetailed(**cached_result)
                else:
                    return ClassifyResponseSimple(**cached_result)
            else:
                # Fallback if cached result format is unexpected
                return ClassifyResponseSimple(
                    label=str(cached_result.get("label", "UNKNOWN")),
                    confidence=float(cached_result.get("confidence", 0.0)),
                    cached=True,
                )

        # Cache miss - call ML API
        logger.info(
            f"Cache miss. Calling ML API for provider: {ml_provider.get_provider_name()}"
        )
        logger.debug(f"Request text length: {len(request.text)} characters")

        # Call ML provider
        raw_response = await ml_provider.classify(request.text)

        # Normalize response
        provider_name = ml_provider.get_provider_name()
        primary_label, primary_confidence, all_labels = normalize_response(
            raw_response, provider_name
        )

        # Prepare response
        response_data = {
            "label": primary_label,
            "confidence": primary_confidence,
            "cached": False,
        }

        # Add detailed labels if detailed format requested
        if settings.response_format == "detailed":
            labels_list = [
                {"label": label, "confidence": conf} for label, conf in all_labels
            ]
            response_data["labels"] = labels_list

        # Store in cache
        try:
            await cache.set(cache_key, response_data, ttl=settings.cache_ttl)
            logger.debug(f"Result cached with key: {cache_key[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

        # Record metrics
        response_time = time.time() - start_time
        metrics.record_request(response_time, cached=False)

        logger.info(
            f"Classification completed: {primary_label} (confidence: {primary_confidence:.4f}) "
            f"in {response_time:.4f}s"
        )

        # Return response
        if settings.response_format == "detailed":
            return ClassifyResponseDetailed(**response_data)
        else:
            return ClassifyResponseSimple(**response_data)

    except httpx.TimeoutException as e:
        response_time = time.time() - start_time
        logger.error(f"Timeout during classification: {e}", exc_info=True)
        metrics.record_error("timeout")
        raise  # Will be handled by exception handler

    except httpx.HTTPStatusError as e:
        response_time = time.time() - start_time
        logger.error(f"HTTP error during classification: {e}", exc_info=True)
        metrics.record_error("api_error")
        raise  # Will be handled by exception handler

    except httpx.RequestError as e:
        response_time = time.time() - start_time
        logger.error(f"Network error during classification: {e}", exc_info=True)
        metrics.record_error("api_error")
        raise  # Will be handled by exception handler

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Unexpected error during classification: {e}", exc_info=True)
        metrics.record_error("internal_error")
        raise  # Will be handled by exception handler


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Checks the availability of:
    - Redis (if using Redis cache)
    - ML API provider

    Returns:
        Health status with details
    """
    health_status = {
        "status": "healthy",
        "details": {},
    }

    # Check cache (Redis if applicable)
    if settings.cache_strategy.value == "redis":
        try:
            if hasattr(cache, "ping"):
                is_alive = await cache.ping()
                if is_alive:
                    health_status["details"]["redis"] = "connected"
                else:
                    health_status["status"] = "unhealthy"
                    health_status["details"]["redis"] = "connection_failed"
            else:
                health_status["details"]["redis"] = "not_available"
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["details"]["redis"] = f"error: {str(e)}"

    # Check ML provider (try a simple test if possible)
    try:
        # Just check if provider is initialized
        if ml_provider is None:
            health_status["status"] = "unhealthy"
            health_status["details"]["ml_provider"] = "not_initialized"
        else:
            provider_name = ml_provider.get_provider_name()
            health_status["details"]["ml_provider"] = f"{provider_name} (initialized)"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["details"]["ml_provider"] = f"error: {str(e)}"

    # Return appropriate status code
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

    return Response(
        content=json.dumps(health_status),
        media_type="application/json",
        status_code=status_code,
    )


@app.get("/metrics")
async def get_metrics():
    """
    Get service metrics.

    Returns:
        JSON with metrics including:
        - Total requests
        - Successful requests
        - Errors by type
        - Cache hits and misses
        - Cache hit rate
        - Average response time
    """
    return metrics.get_metrics()

