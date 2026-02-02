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
from api.models.llm_models import AssistantRequest, error_response_body
from api.services.llm_service import get_llm_service, LLMService
from api.siu_client import SiuClient
from api.exceptions import ServiceError


router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])


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
        context["chat_messages"] = load_chat_history(siu_client, request.chat_history_irv_id)

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
                content=response,
            )

        # Если ответ успешен, сохраняем историю чата
        messages_sent = context.get("messages_sent") or []
        # В историю сохраняем только answer (без reasoning), который уже извлечён в response.content
        full_messages = messages_sent + [
            {"role": "assistant", "content": response.get("content", "") or ""},
        ]
        chat_history_result = None
        try:
            chat_history_result = save_chat_history(
                siu_client,
                chat_history_irv_id=request.chat_history_irv_id,
                irv_id=request.irv_id,
                chat_title=response.get("chat_title"),
                chat_summary=response.get("chat_summary"),
                full_messages=full_messages,
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
