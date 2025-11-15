"""ML Providers package for different ML API integrations."""

from typing import Optional

from config import settings, MLProvider as MLProviderEnum

from .base import MLProvider
from .local import LocalMLProvider
from .huggingface import HuggingFaceProvider
from .openai import OpenAIProvider


def create_provider(config: Optional[object] = None) -> MLProvider:
    """
    Factory function to create ML provider based on configuration.

    Args:
        config: Optional settings object (defaults to global settings)

    Returns:
        Instance of ML provider

    Raises:
        ValueError: If provider type is not supported or configuration is invalid
    """
    if config is None:
        config = settings

    provider_type = config.ml_provider
    timeout = config.api_timeout

    if provider_type == MLProviderEnum.LOCAL:
        if not config.local_model_url:
            raise ValueError("local_model_url is required for local provider")
        return LocalMLProvider(
            model_url=config.local_model_url,
            timeout=timeout,
        )

    elif provider_type == MLProviderEnum.HUGGINGFACE:
        if not config.huggingface_api_key:
            raise ValueError("huggingface_api_key is required for Hugging Face provider")
        if not config.huggingface_model:
            raise ValueError("huggingface_model is required for Hugging Face provider")
        return HuggingFaceProvider(
            api_key=config.huggingface_api_key,
            model_name=config.huggingface_model,
            api_url=config.huggingface_api_url,
            timeout=timeout,
        )

    elif provider_type == MLProviderEnum.OPENAI:
        if not config.openai_api_key:
            raise ValueError("openai_api_key is required for OpenAI provider")
        return OpenAIProvider(
            api_key=config.openai_api_key,
            model=config.openai_model or "gpt-3.5-turbo",
            timeout=timeout,
        )

    else:
        raise ValueError(f"Unsupported ML provider: {provider_type}")


__all__ = [
    "MLProvider",
    "LocalMLProvider",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "create_provider",
]
