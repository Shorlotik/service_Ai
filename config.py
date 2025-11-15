"""Configuration module for loading and validating environment variables."""

from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class MLProvider(str, Enum):
    """Supported ML providers."""

    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class CacheStrategy(str, Enum):
    """Supported cache strategies."""

    MEMORY = "memory"
    REDIS = "redis"


class LogFormat(str, Enum):
    """Supported log formats."""

    SIMPLE = "simple"
    JSON = "json"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ML Provider configuration
    ml_provider: MLProvider = Field(
        default=MLProvider.LOCAL,
        description="ML provider to use (local, huggingface, openai)",
    )

    # Local model configuration
    local_model_url: Optional[str] = Field(
        default=None,
        description="URL for local ML model endpoint",
    )

    # Hugging Face configuration
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key",
    )
    huggingface_model: Optional[str] = Field(
        default=None,
        description="Hugging Face model name",
    )
    huggingface_api_url: str = Field(
        default="https://api-inference.huggingface.co/models",
        description="Hugging Face API base URL",
    )

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_model: Optional[str] = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model name",
    )

    # Cache configuration
    cache_strategy: CacheStrategy = Field(
        default=CacheStrategy.MEMORY,
        description="Cache strategy (memory, redis)",
    )
    cache_ttl: int = Field(
        default=86400,
        description="Cache TTL in seconds (default: 24 hours)",
        ge=1,
    )

    # Redis configuration
    redis_host: str = Field(
        default="localhost",
        description="Redis host",
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port",
        ge=1,
        le=65535,
    )
    redis_db: int = Field(
        default=0,
        description="Redis database number",
        ge=0,
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password",
    )

    # API configuration
    api_timeout: int = Field(
        default=30,
        description="Timeout for ML API requests in seconds",
        ge=1,
    )
    max_text_length: int = Field(
        default=10000,
        description="Maximum text length for classification",
        ge=1,
    )

    # Response format
    response_format: str = Field(
        default="simple",
        description="Response format (simple, detailed)",
    )

    # Logging configuration
    log_format: LogFormat = Field(
        default=LogFormat.SIMPLE,
        description="Log format (simple, json)",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Application configuration
    app_name: str = Field(
        default="ML API Wrapper Service",
        description="Application name",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )

    @model_validator(mode="after")
    def validate_provider_config(self):
        """Validate provider-specific configuration."""
        if self.ml_provider == MLProvider.LOCAL:
            if not self.local_model_url:
                raise ValueError(
                    "local_model_url is required when ml_provider is 'local'"
                )
        elif self.ml_provider == MLProvider.HUGGINGFACE:
            if not self.huggingface_api_key:
                raise ValueError(
                    "huggingface_api_key is required when ml_provider is 'huggingface'"
                )
            if not self.huggingface_model:
                raise ValueError(
                    "huggingface_model is required when ml_provider is 'huggingface'"
                )
        elif self.ml_provider == MLProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "openai_api_key is required when ml_provider is 'openai'"
                )
        return self

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow reading from environment variables with different naming conventions
        env_prefix = ""


# Create global settings instance
settings = Settings()

