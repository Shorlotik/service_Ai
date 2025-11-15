"""Normalizers for converting different ML provider responses to unified format."""

from typing import Any, Dict, List, Tuple


def normalize_response(
    raw_response: Dict[str, Any], provider_name: str
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Normalize ML provider response to unified format.

    Args:
        raw_response: Raw response from ML provider
        provider_name: Name of the provider (for format detection)

    Returns:
        Tuple of:
        - Primary label (str)
        - Primary confidence (float)
        - List of all (label, confidence) tuples sorted by confidence descending

    Raises:
        ValueError: If response format is not recognized or invalid
    """
    if provider_name == "huggingface":
        return _normalize_huggingface(raw_response)
    elif provider_name == "openai":
        return _normalize_openai(raw_response)
    elif provider_name == "local":
        return _normalize_local(raw_response)
    else:
        # Try to auto-detect format
        return _normalize_auto(raw_response)


def _normalize_huggingface(response: Dict[str, Any]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Normalize Hugging Face API response.

    Expected format:
    [
        {"label": "POSITIVE", "score": 0.95},
        {"label": "NEGATIVE", "score": 0.05}
    ]
    or
    [{"label": "LABEL", "score": 0.99}]
    """
    if not isinstance(response, list):
        raise ValueError(f"Expected list from Hugging Face, got {type(response)}")

    labels_scores = []
    for item in response:
        if not isinstance(item, dict):
            continue
        label = item.get("label", "")
        score = item.get("score", 0.0)
        if label and isinstance(score, (int, float)):
            labels_scores.append((str(label), float(score)))

    if not labels_scores:
        raise ValueError("No valid labels found in Hugging Face response")

    # Sort by confidence descending
    labels_scores.sort(key=lambda x: x[1], reverse=True)

    primary_label, primary_confidence = labels_scores[0]
    return primary_label, primary_confidence, labels_scores


def _normalize_openai(response: Dict[str, Any]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Normalize OpenAI API response.

    Expected format (for classification):
    {
        "choices": [{
            "message": {
                "content": "POSITIVE" or JSON with labels
            }
        }]
    }
    or custom format from local wrapper
    """
    # OpenAI responses can vary, try to extract from common patterns
    if "choices" in response:
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            content = choices[0].get("message", {}).get("content", "")
            # If content is a label string, return it with default confidence
            if isinstance(content, str) and content.strip():
                label = content.strip()
                return label, 1.0, [(label, 1.0)]

    # Try to find labels in response
    if "label" in response:
        label = str(response.get("label", ""))
        confidence = float(response.get("confidence", response.get("score", 1.0)))
        return label, confidence, [(label, confidence)]

    raise ValueError("Could not normalize OpenAI response format")


def _normalize_local(response: Dict[str, Any]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Normalize local model response.

    Expected formats:
    1. {"label": "POSITIVE", "confidence": 0.95}
    2. {"labels": [{"label": "POS", "score": 0.9}, {"label": "NEG", "score": 0.1}]}
    3. [{"label": "LABEL", "score": 0.99}]
    """
    # Format 1: Simple label + confidence
    if "label" in response and "confidence" in response:
        label = str(response["label"])
        confidence = float(response["confidence"])
        return label, confidence, [(label, confidence)]

    # Format 2: Multiple labels
    if "labels" in response:
        labels_list = response["labels"]
        if isinstance(labels_list, list):
            labels_scores = []
            for item in labels_list:
                if isinstance(item, dict):
                    label = item.get("label", item.get("name", ""))
                    score = item.get("score", item.get("confidence", 0.0))
                    if label and isinstance(score, (int, float)):
                        labels_scores.append((str(label), float(score)))

            if labels_scores:
                labels_scores.sort(key=lambda x: x[1], reverse=True)
                primary_label, primary_confidence = labels_scores[0]
                return primary_label, primary_confidence, labels_scores

    # Format 3: List format (similar to Hugging Face)
    if isinstance(response, list):
        return _normalize_huggingface(response)

    # Try to extract from common fields
    if "prediction" in response:
        pred = response["prediction"]
        if isinstance(pred, dict):
            label = pred.get("label", pred.get("class", ""))
            confidence = float(pred.get("confidence", pred.get("score", 1.0)))
            if label:
                return str(label), confidence, [(str(label), confidence)]

    raise ValueError("Could not normalize local model response format")


def _normalize_auto(response: Dict[str, Any]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Auto-detect and normalize response format.

    Tries common patterns to extract labels and confidences.
    """
    # Try Hugging Face format first (most common)
    if isinstance(response, list):
        try:
            return _normalize_huggingface(response)
        except (ValueError, KeyError, TypeError):
            pass

    # Try simple dict format
    if isinstance(response, dict):
        # Try local format
        try:
            return _normalize_local(response)
        except (ValueError, KeyError, TypeError):
            pass

        # Try OpenAI format
        try:
            return _normalize_openai(response)
        except (ValueError, KeyError, TypeError):
            pass

    raise ValueError(
        f"Could not auto-detect response format. Response type: {type(response)}"
    )

