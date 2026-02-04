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
    max_retry_count: int = Field(3, description="Максимальное количество повторных попыток при отсутствии answer в ответе")

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
    chat_history_irv_id: Optional[str] = Field(
        default=None,
        description="""UUID версии информационного объекта с файлом, содержащим историю чата. 
         Если не пусто — используется заданная история чата, system_prompt игнорируется;
         иначе используется system_prompt"""
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
    file_irv_ids: Optional[List[str]] = Field(None, description="Идентификаторы версий информационных объектов со входными файлами")
    internet: bool = Field(False, description="Использовать интернет")
    knowledge_base: bool = Field(False, description="Использовать базу знаний")

    # Контекст
    irv_id: Optional[str] = Field(None, description="UUID версии текущего информационного объекта")


class AssistantResponse(BaseModel):
    """Ответ от LLM."""

    content: str = Field(..., description="Сгенерированный текст")
    model: str = Field(..., description="Использованная модель")
    usage: Optional[Dict[str, Any]] = Field(None, description="Информация об использовании токенов")
    finish_reason: Optional[str] = Field(None, description="Причина завершения генерации")
    chat_title: Optional[str] = Field(None, description="Заголовок диалога (из структурированного ответа LLM)")
    chat_summary: Optional[str] = Field(None, description="Краткое описание диалога (из структурированного ответа LLM)")


class ValidationErrorItem(BaseModel):
    """Одна ошибка валидации поля."""
    
    field: str = Field(..., description="Путь к полю (например: body.current_message)")
    message: str = Field(..., description="Сообщение об ошибке")
    type: Optional[str] = Field(None, description="Тип ошибки Pydantic (например: value_error.missing)")


class RAGRequest(BaseModel):
    """Запрос для работы с RAG (добавление/удаление файлов из векторной БД)."""
    
    vdb_url: str = Field(..., description="URL векторной базы данных")
    irv_id: str = Field(..., description="Идентификатор версии информационного объекта")
    action: str = Field(..., description="Действие: 'add' - добавить файлы, 'remove' - удалить файлы")
    
    # Параметры эмбеддингов (опционально, если не указаны - используются из конфигурации)
    embed_api_key: Optional[str] = Field(None, description="API ключ для эмбеддингов")
    embed_url: Optional[str] = Field(None, description="URL API эмбеддингов")
    embed_model_name: Optional[str] = Field(None, description="Модель эмбеддингов")


class RAGAddResponse(BaseModel):
    """Ответ при добавлении файлов в RAG."""
    
    success: bool = Field(..., description="Успешность операции")
    irv_id: str = Field(..., description="Идентификатор версии ИО")
    files_processed: int = Field(..., description="Количество обработанных файлов")
    chunks_saved: int = Field(..., description="Количество сохраненных чанков")
    toc_chunks_saved: int = Field(0, description="Количество сохраненных чанков оглавления")
    files_info: List[Dict[str, Any]] = Field(default_factory=list, description="Информация о каждом обработанном файле")


class RAGRemoveResponse(BaseModel):
    """Ответ при удалении файлов из RAG."""
    
    success: bool = Field(..., description="Успешность операции")
    irv_id: str = Field(..., description="Идентификатор версии ИО")
    chunks_deleted: int = Field(..., description="Количество удаленных чанков")


class CollectionListRequest(BaseModel):
    """Запрос для получения списка коллекций в векторной БД."""
    
    vdb_url: str = Field(..., description="URL векторной базы данных")


class CollectionInfo(BaseModel):
    """Информация об одной коллекции."""
    
    name: str = Field(..., description="Имя коллекции")
    points_count: int = Field(..., description="Количество точек в коллекции")
    status: Optional[str] = Field(None, description="Статус коллекции")
    vector_size: Optional[int] = Field(None, description="Размер вектора")
    distance: Optional[str] = Field(None, description="Метрика расстояния")


class CollectionListResponse(BaseModel):
    """Ответ со списком коллекций."""
    
    success: bool = Field(..., description="Успешность операции")
    collections: List[CollectionInfo] = Field(default_factory=list, description="Список коллекций")
    total: int = Field(..., description="Общее количество коллекций")


class CollectionDeleteRequest(BaseModel):
    """Запрос для удаления коллекции из векторной БД."""
    
    vdb_url: str = Field(..., description="URL векторной базы данных")
    collection_name: str = Field(..., description="Имя коллекции для удаления")


class CollectionDeleteResponse(BaseModel):
    """Ответ при удалении коллекции."""
    
    success: bool = Field(..., description="Успешность операции")
    collection_name: str = Field(..., description="Имя удаленной коллекции")
    message: str = Field(..., description="Сообщение о результате операции")


class QdrantHealthCheckRequest(BaseModel):
    """Запрос для проверки доступности Qdrant."""
    
    vdb_url: str = Field(..., description="URL векторной базы данных")


class QdrantHealthCheckResponse(BaseModel):
    """Ответ при проверке доступности Qdrant."""
    
    available: bool = Field(..., description="Доступен ли Qdrant сервер")
    message: Optional[str] = Field(None, description="Сообщение о статусе")
    version: Optional[str] = Field(None, description="Версия Qdrant (если доступна)")


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
