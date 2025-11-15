"""Pydantic models for request and response validation."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    """Request model for text classification."""

    text: str = Field(
        ...,
        description="Text to classify",
        min_length=1,
        max_length=10000,
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and normalize text."""
        text = v.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        return text


class LabelConfidence(BaseModel):
    """Model for label with confidence score."""

    label: str = Field(..., description="Classification label")
    confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class ClassifyResponseSimple(BaseModel):
    """Simple response model for classification."""

    label: str = Field(..., description="Primary classification label")
    confidence: float = Field(
        ...,
        description="Confidence score for the primary label (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    cached: bool = Field(
        ...,
        description="Whether the result was retrieved from cache",
    )


class ClassifyResponseDetailed(BaseModel):
    """Detailed response model for classification with all labels."""

    label: str = Field(..., description="Primary classification label")
    confidence: float = Field(
        ...,
        description="Confidence score for the primary label (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    labels: List[LabelConfidence] = Field(
        ...,
        description="All classification labels with confidence scores",
        min_length=1,
    )
    cached: bool = Field(
        ...,
        description="Whether the result was retrieved from cache",
    )


# Union type for response (can be simple or detailed)
ClassifyResponse = Union[ClassifyResponseSimple, ClassifyResponseDetailed]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(
        ...,
        description="Error type (timeout, api_error, validation_error, internal_error)",
    )
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(
        None,
        description="Additional error details (optional)",
    )

