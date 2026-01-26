"""
API models модуль.
"""

from api.models.llm_models import (
    OpenAIConfig,
    LLMRequest,
    LLMResponse,
    ErrorResponse
)

__all__ = [
    "OpenAIConfig",
    "LLMRequest",
    "LLMResponse",
    "ErrorResponse"
]

