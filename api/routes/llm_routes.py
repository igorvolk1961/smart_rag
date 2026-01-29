"""
Роуты для работы с LLM API.
"""

import json
import re
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from api.models.llm_models import AssistantRequest, error_response_body
from api.services.llm_service import get_llm_service, LLMService
from api.exceptions import ServiceError


router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])


def _normalize_content_to_response(content: str):
    """
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
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Генерация ответа ассистента. Всегда HTTP 200: проверяйте наличие поля error в теле.
    Возвращает только content; при JSON-обёртке — распарсенный объект.
    """
    try:
        logger.info("Получен запрос на генерацию ответа от LLM")

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
            response = await llm_service.simple_llm_call(request)
        else:
            response = await llm_service.agent_call(request)

        logger.info("Ответ успешно сгенерирован")
        body = _normalize_content_to_response(response.content)
        return JSONResponse(status_code=200, content=body)

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
