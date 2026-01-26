"""
Модели для работы с LLM API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """Конфигурация для доступа к OpenAI API."""
    
    api_key: str = Field(..., description="API ключ OpenAI")
    base_url: Optional[str] = Field(None, description="Базовый URL API (для совместимости с другими провайдерами)")
    organization: Optional[str] = Field(None, description="ID организации OpenAI")
    timeout: Optional[float] = Field(None, description="Таймаут запроса в секундах")
    max_retries: Optional[int] = Field(None, description="Максимальное количество повторных попыток")


class LLMRequest(BaseModel):
    """Базовый запрос к LLM."""
    
    # Конфигурация OpenAI будет передаваться в каждом запросе
    openai_config: OpenAIConfig = Field(..., description="Конфигурация доступа к OpenAI API")
    
    # Пока базовая структура, детали будут добавлены позже
    message: Optional[str] = Field(None, description="Текст запроса")
    
    # Дополнительные параметры запроса
    model: Optional[str] = Field("gpt-3.5-turbo", description="Модель для использования")
    temperature: Optional[float] = Field(0.7, description="Температура генерации")
    max_tokens: Optional[int] = Field(None, description="Максимальное количество токенов")
    
    # Дополнительные поля для расширения в будущем
    extra_params: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры запроса")


class LLMResponse(BaseModel):
    """Ответ от LLM."""
    
    content: str = Field(..., description="Сгенерированный текст")
    model: str = Field(..., description="Использованная модель")
    usage: Optional[Dict[str, Any]] = Field(None, description="Информация об использовании токенов")
    finish_reason: Optional[str] = Field(None, description="Причина завершения генерации")


class ErrorResponse(BaseModel):
    """Модель для ошибок API."""
    
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Детали ошибки")
