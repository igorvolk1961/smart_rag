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
    """Базовый запрос к LLM (внутреннее использование)."""
    
    openai_config: OpenAIConfig = Field(..., description="Конфигурация доступа к OpenAI API")
    model: Optional[str] = Field("gpt-3.5-turbo", description="Модель для использования")
    temperature: Optional[float] = Field(0.7, description="Температура генерации")
    max_tokens: Optional[int] = Field(None, description="Максимальное количество токенов")

    # Основной параметр: список сообщений [{"role": "user"|"assistant"|"system", "content": "..."}]
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Сообщения чата для контекста и запроса")

    extra_params: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры (n и т.д.)")


class ChatMessage(BaseModel):
    """Одно сообщение в истории чата."""
    role: str = Field(..., description="Роль: user, assistant или system")
    content: str = Field(..., description="Текст сообщения")


class AssistantRequest(BaseModel):
    """Запрос к ассистенту (тело эндпоинта /generate)."""
    
    current_message: str = Field(..., description="Текущее сообщение пользователя")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="История чата. Если не пуста — используется она, system_prompt игнорируется; иначе используется system_prompt"
    )
    system_prompt: Optional[str] = Field(None, description="Системный промпт (используется только при пустой истории чата)")
    temperature: float = Field(0.2, description="Температура генерации")
    max_tokens: int = Field(8000, description="Максимальное количество токенов")
    n: int = Field(1, description="Количество вариантов ответа")
    
    # LLM
    llm_api_key: str = Field(..., description="API ключ LLM")
    llm_auth_type: str = Field("0", description="Тип авторизации LLM")
    llm_model_name: str = Field(..., description="Модель LLM")
    llm_url: str = Field(..., description="URL LLM API")
    
    # Embeddings
    embed_api_key: Optional[str] = Field(None, description="API ключ для эмбеддингов")
    embed_auth_type: Optional[str] = Field(None, description="Тип авторизации эмбеддингов")
    embed_model_name: Optional[str] = Field(None, description="Модель эмбеддингов")
    embed_url: Optional[str] = Field(None, description="URL API эмбеддингов")
    
    # Поиск и БД
    search_api_key: Optional[str] = Field(None, description="API ключ поиска")
    search_url: Optional[str] = Field(None, description="URL поиска")
    vdb_url: Optional[str] = Field(None, description="URL векторной БД")
    
    # Режимы
    files: Optional[List[str]] = Field("", description="Файлы")
    internet: bool = Field(False, description="Использовать интернет")
    knowledge_base: bool = Field(False, description="Использовать базу знаний")


class AssistantResponse(BaseModel):
    """Ответ от LLM."""
    
    content: str = Field(..., description="Сгенерированный текст")
    model: str = Field(..., description="Использованная модель")
    usage: Optional[Dict[str, Any]] = Field(None, description="Информация об использовании токенов")
    finish_reason: Optional[str] = Field(None, description="Причина завершения генерации")


class ValidationErrorItem(BaseModel):
    """Одна ошибка валидации поля."""
    
    field: str = Field(..., description="Путь к полю (например: body.current_message)")
    message: str = Field(..., description="Сообщение об ошибке")
    type: Optional[str] = Field(None, description="Тип ошибки Pydantic (например: value_error.missing)")


class ErrorResponse(BaseModel):
    """Модель для ошибок API."""
    
    error: str = Field(..., description="Краткое описание ошибки (тип/категория)")
    detail: Optional[str] = Field(None, description="Подробное сообщение для отладки")
    code: Optional[str] = Field(None, description="Код ошибки (машиночитаемый)")
    errors: Optional[List[ValidationErrorItem]] = Field(
        None,
        description="Список ошибок по полям (для validation_error)",
    )


def error_response_body(
    error: str,
    detail: Optional[str] = None,
    code: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """Тело ответа об ошибке в формате ErrorResponse."""
    payload: Dict[str, Any] = {
        "error": error,
        "detail": detail or error,
    }
    if code is not None:
        payload["code"] = code
    if errors is not None:
        payload["errors"] = errors
    return payload
