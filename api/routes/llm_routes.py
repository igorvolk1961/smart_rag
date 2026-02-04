"""
Роуты для работы с LLM API.
"""

import json
import re
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.chat_history import load_chat_history, save_chat_history
from api.models.llm_models import (
    AssistantRequest, 
    error_response_body, 
    RAGRequest,
    CollectionListRequest,
    CollectionDeleteRequest,
    QdrantHealthCheckRequest,
    QdrantHealthCheckResponse,
)
from api.services.llm_service import get_llm_service, LLMService
from api.services.rag_service import RAGService
from api.siu_client import SiuClient
from api.exceptions import ServiceError


router = APIRouter(prefix="/v1", tags=["LLM"])


def extract_callback_info(http_request: Request) -> dict[str, Any]:
    """
    Извлекает из заголовков referer и cookie base_referer_url и JSESSIONID.
    Возвращает объект context с полями base_referer_url и jsessionid.
    """
    referer = http_request.headers.get("referer") or ""
    cookie = http_request.headers.get("cookie") or ""

    base_referer_url = ""
    if referer:
        parsed = urlparse(referer)
        base_referer_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else referer

    jsessionid = None
    for part in cookie.split(";"):
        part = part.strip()
        if part.upper().startswith("JSESSIONID="):
            jsessionid = part.split("=", 1)[1].strip()
            break

    return base_referer_url, jsessionid


def _normalize_content_to_response(content: str):
    """user_post
    Возвращает тело ответа: только content.
    Если content — текст json-объекта в обёртке ```json ... ``` или после strip
    начинается с '{', возвращается распарсенный JSON-объект; иначе — строка content.
    """
    if not content or not content.strip():
        return content
    text = content.strip()
    # Удалить префикс ```json (без учёта регистра) и финальные ```
    json_block = re.match(r"^```json\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if json_block:
        inner = json_block.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    # Текст, начинающийся с {, тоже пробуем распарсить как JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return content


@router.post(
    "/generate",
    summary="Генерация ответа от LLM",
    description=(
        "При успехе возвращает 200 и только поле content ответа. "
        "Если content — JSON в обёртке ```json ... ```, возвращается распарсенный объект. "
        "При internet=false и knowledge_base=false вызывается простой LLM; иначе — agent_call. "
        "При ошибке — 200 и объект с полем error (и detail, code, errors при валидации)."
    ),
)
async def generate_response(
    request: AssistantRequest,
    http_request: Request,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Генерация ответа ассистента. Всегда HTTP 200: проверяйте наличие поля error в теле.
    Возвращает только content; при JSON-обёртке — распарсенный объект.
    """
    try:
        logger.info("Получен запрос на генерацию ответа от LLM")

        base_referer_url, jsessionid = extract_callback_info(http_request)
        siu_client = SiuClient(base_referer_url, jsessionid)
        context = {}
        context["userInfo"] = siu_client.get_current_user_info()
        if request.irv_id:
            context["irvInfo"] = siu_client.get_irv_info(request.irv_id)
        else:
            context["irvInfo"] = {}
        chat_messages, irv_exists = load_chat_history(siu_client, request.chat_history_irv_id)
        context["chat_messages"] = chat_messages
        context["chat_history_irv_exists"] = irv_exists

        if not request.current_message or not request.current_message.strip():
            return JSONResponse(
                status_code=200,
                content=error_response_body(
                    error="Ошибка валидации запроса",
                    detail="Поле 'current_message' обязательно для заполнения и не может быть пустым.",
                    code="missing_current_message",
                ),
            )

        if not request.internet and not request.knowledge_base:
            response = await llm_service.simple_llm_call(request, context)
        else:
            response = await llm_service.agent_call(request, context)

        # Проверяем, есть ли ошибка в ответе
        if "error" in response:
            return JSONResponse(
                status_code=200,
                content=response,  # response уже содержит поле error
            )

        # Если ответ успешен, сохраняем историю чата
        # Формируем полную историю: загруженная история + текущее сообщение + ответ ассистента
        chat_messages = context.get("chat_messages") or []
        current_message = {"role": "user", "content": request.current_message}
        assistant_message = {"role": "assistant", "content": response.get("content", "") or ""}
        # В историю сохраняем только answer (без reasoning), который уже извлечён в response.content
        full_messages = chat_messages + [current_message, assistant_message]
        chat_history_result = None
        try:
            chat_history_result = save_chat_history(
                siu_client,
                chat_history_irv_id=request.chat_history_irv_id,
                irv_id=request.irv_id,
                chat_title=response.get("chat_title"),
                chat_summary=response.get("chat_summary"),
                full_messages=full_messages,
                irv_exists=context.get("chat_history_irv_exists", False),
                has_messages=bool(context.get("chat_messages")),
            )
        except ServiceError as e:
            logger.warning("Сохранение истории чата не выполнено: {} {}", e.error, e.detail)
        except Exception as e:
            logger.exception("Ошибка при сохранении истории чата: {}", e)

        logger.info("Ответ успешно сгенерирован")
        # Формируем ответ с content и опционально chat_history
        body = _normalize_content_to_response(response.get("content", ""))
        if isinstance(body, dict):
            if chat_history_result:
                body["chat_history"] = chat_history_result
        else:
            # Если body - строка, создаём словарь
            result_dict = {"content": body}
            if chat_history_result:
                result_dict["chat_history"] = chat_history_result
            body = result_dict
        jsonResponse = JSONResponse(status_code=200, content=body)
        return jsonResponse

    except ServiceError as e:
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error=e.error,
                detail=e.detail,
                code=e.code,
            ),
        )
    except Exception as e:
        logger.exception("Неожиданная ошибка при генерации ответа")
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error="Внутренняя ошибка сервера",
                detail=str(e),
                code="internal_error",
            ),
        )


@router.get(
    "/cache/info",
    summary="Информация о кэше",
    description="Получение информации о размере кэша конфигураций"
)
async def get_cache_info(
    llm_service: LLMService = Depends(get_llm_service)
) -> dict:
    """
    Получение информации о кэше конфигураций.
    
    Args:
        llm_service: Сервис для работы с LLM
        
    Returns:
        Информация о кэше
    """
    cache_size = llm_service.cache.get_cache_size()
    return {
        "cache_size": cache_size,
        "message": f"В кэше находится {cache_size} конфигураций"
    }


@router.delete(
    "/cache/clear",
    summary="Очистка кэша",
    description="Очистка кэша конфигураций OpenAI API"
)
async def clear_cache(
    llm_service: LLMService = Depends(get_llm_service)
) -> dict:
    """
    Очистка кэша конфигураций.
    
    Args:
        llm_service: Сервис для работы с LLM
        
    Returns:
        Результат операции
    """
    llm_service.cache.clear_cache()
    logger.info("Кэш конфигураций очищен")
    return {
        "message": "Кэш успешно очищен"
    }


@router.post(
    "/rag/manage",
    summary="Управление файлами в RAG",
    description=(
        "Добавление или удаление файлов информационного объекта в векторную базу данных. "
        "В режиме 'add' файлы docx и txt обрабатываются через SmartChanker и сохраняются в БД. "
        "В режиме 'remove' удаляются все чанки, связанные с указанным irv_id."
    ),
)
async def manage_rag_files(
    request: RAGRequest,
    http_request: Request,
) -> JSONResponse:
    """
    Управление файлами в RAG-системе.
    
    Args:
        request: Запрос с параметрами vdb_url, irv_id и action
        http_request: HTTP запрос для извлечения контекста
        
    Returns:
        Результат операции (добавление или удаление)
    """
    try:
        logger.info(f"Получен запрос на управление RAG файлами: action={request.action}, irv_id={request.irv_id}")
        
        # Валидация action
        if request.action not in ["add", "remove"]:
            return JSONResponse(
                status_code=200,
                content=error_response_body(
                    error="Ошибка валидации запроса",
                    detail=f"Недопустимое значение action: '{request.action}'. Допустимые значения: 'add', 'remove'.",
                    code="invalid_action",
                ),
            )
        
        # Извлечение контекста для работы с СИУ
        base_referer_url, jsessionid = extract_callback_info(http_request)
        siu_client = SiuClient(base_referer_url, jsessionid)
        
        # Инициализация RAG сервиса
        rag_service = RAGService()
        
        if request.action == "add":
            result = rag_service.add_files_to_rag(request, siu_client)
            # Убираем поле success, если оно есть, так как успех определяется наличием content
            if isinstance(result, dict) and "success" in result:
                result = {k: v for k, v in result.items() if k != "success"}
            return JSONResponse(status_code=200, content={"content": result})
        else:  # remove
            result = rag_service.remove_files_from_rag(request)
            # Убираем поле success, если оно есть
            if isinstance(result, dict) and "success" in result:
                result = {k: v for k, v in result.items() if k != "success"}
            return JSONResponse(status_code=200, content={"content": result})
            
    except ServiceError as e:
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error=e.error,
                detail=e.detail,
                code=e.code,
            ),
        )
    except Exception as e:
        logger.exception("Неожиданная ошибка при управлении RAG файлами")
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error="Внутренняя ошибка сервера",
                detail=str(e),
                code="internal_error",
            ),
        )


@router.post(
    "/rag/health",
    summary="Проверка доступности Qdrant",
    description="Быстрая проверка доступности Qdrant сервера без выполнения операций с данными"
)
async def check_qdrant_health(
    request: QdrantHealthCheckRequest,
) -> JSONResponse:
    """
    Проверка доступности Qdrant сервера.
    
    Args:
        request: Запрос с параметром vdb_url (тело запроса)
        
    Returns:
        Статус доступности Qdrant
    """
    try:
        logger.info(f"Получен запрос на проверку доступности Qdrant: vdb_url={request.vdb_url}")
        
        from rag.vector_store import QdrantVectorStoreManager
        from utils.config import get_config
        
        # Загрузка конфигурации
        config = get_config()
        qdrant_config = config.get("qdrant", {})
        
        # Инициализация векторного хранилища
        vdb_url = request.vdb_url.strip().rstrip("/")
        if not vdb_url.startswith("http"):
            vdb_url = f"http://{vdb_url}"
        
        # Создаем временный менеджер для проверки подключения
        vector_store_manager = QdrantVectorStoreManager(
            url=vdb_url,
            api_key=qdrant_config.get("api_key"),
            collection_name="temp",  # Временное имя, не используется
            vector_size=qdrant_config.get("vector_size", 1024),
            timeout=5  # Короткий таймаут для быстрой проверки
        )
        
        # Проверяем подключение
        available, error_message = vector_store_manager.check_connection(timeout=5)
        
        if available:
            # Пытаемся получить версию Qdrant
            version = None
            try:
                import httpx
                response = httpx.get(f"{vdb_url}/", timeout=5)
                if response.status_code == 200:
                    info = response.json()
                    version = info.get("version")
            except:
                pass
            
            result = QdrantHealthCheckResponse(
                available=True,
                message="Qdrant сервер доступен",
                version=version
            )
        else:
            result = QdrantHealthCheckResponse(
                available=False,
                message=error_message or "Qdrant сервер недоступен"
            )
        
        return JSONResponse(status_code=200, content=result.model_dump())
        
    except Exception as e:
        logger.exception("Неожиданная ошибка при проверке доступности Qdrant")
        return JSONResponse(
            status_code=200,
            content=QdrantHealthCheckResponse(
                available=False,
                message=f"Ошибка при проверке доступности: {str(e)}"
            ).model_dump()
        )


@router.post(
    "/rag/collections",
    summary="Получение списка коллекций",
    description="Получение информации о всех коллекциях в векторной базе данных"
)
async def get_collections(
    request: CollectionListRequest,
) -> JSONResponse:
    """
    Получение списка коллекций в векторной БД.
    
    Args:
        request: Запрос с параметром vdb_url (тело запроса)
        
    Returns:
        Список коллекций с информацией о каждой
    """
    try:
        logger.info(f"Получен запрос на получение списка коллекций: vdb_url={request.vdb_url}")
        
        # Инициализация RAG сервиса
        rag_service = RAGService()
        
        result = rag_service.get_collections(request)
        
        # Убираем поле success, если оно есть, так как успех определяется наличием content
        if isinstance(result, dict) and "success" in result:
            result = {k: v for k, v in result.items() if k != "success"}
        
        return JSONResponse(status_code=200, content={"content": result})
        
    except ServiceError as e:
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error=e.error,
                detail=e.detail,
                code=e.code,
            ),
        )
    except Exception as e:
        logger.exception("Неожиданная ошибка при получении списка коллекций")
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error="Внутренняя ошибка сервера",
                detail=str(e),
                code="internal_error",
            ),
        )


@router.delete(
    "/rag/collections/{collection_name}",
    summary="Удаление коллекции",
    description="Удаление указанной коллекции из векторной базы данных"
)
async def delete_collection(
    collection_name: str,
    vdb_url: str,
) -> JSONResponse:
    """
    Удаление коллекции из векторной БД.
    
    Args:
        collection_name: Имя коллекции для удаления (path parameter)
        vdb_url: URL векторной базы данных (query parameter)
        
    Returns:
        Результат удаления коллекции
    """
    try:
        logger.info(f"Получен запрос на удаление коллекции: collection_name={collection_name}, vdb_url={vdb_url}")
        
        # Валидация параметров
        if not collection_name or not collection_name.strip():
            return JSONResponse(
                status_code=200,
                content=error_response_body(
                    error="Ошибка валидации запроса",
                    detail="Параметр 'collection_name' обязателен для заполнения",
                    code="missing_collection_name",
                ),
            )
        
        if not vdb_url or not vdb_url.strip():
            return JSONResponse(
                status_code=200,
                content=error_response_body(
                    error="Ошибка валидации запроса",
                    detail="Параметр 'vdb_url' обязателен для заполнения",
                    code="missing_vdb_url",
                ),
            )
        
        # Инициализация RAG сервиса
        rag_service = RAGService()
        
        request = CollectionDeleteRequest(
            vdb_url=vdb_url.strip(),
            collection_name=collection_name.strip()
        )
        result = rag_service.delete_collection(request)
        
        # Убираем поле success, если оно есть, так как успех определяется наличием content
        if isinstance(result, dict) and "success" in result:
            result = {k: v for k, v in result.items() if k != "success"}
        
        return JSONResponse(status_code=200, content={"content": result})
        
    except ServiceError as e:
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error=e.error,
                detail=e.detail,
                code=e.code,
            ),
        )
    except Exception as e:
        logger.exception("Неожиданная ошибка при удалении коллекции")
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error="Внутренняя ошибка сервера",
                detail=str(e),
                code="internal_error",
            ),
        )


