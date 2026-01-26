"""
Роуты для работы с LLM API.
"""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from api.models.llm_models import LLMRequest, LLMResponse, ErrorResponse
from api.services.llm_service import get_llm_service, LLMService


router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])


@router.post(
    "/generate",
    response_model=LLMResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Генерация ответа от LLM",
    description="Отправка запроса к LLM через OpenAI API с кэшированием конфигурации"
)
async def generate_response(
    request: LLMRequest,
    llm_service: LLMService = Depends(get_llm_service)
) -> LLMResponse:
    """
    Генерация ответа от LLM.
    
    Args:
        request: Запрос к LLM с конфигурацией OpenAI API
        llm_service: Сервис для работы с LLM
        
    Returns:
        Ответ от LLM
        
    Raises:
        HTTPException: При ошибке обработки запроса
    """
    try:
        logger.info("Получен запрос на генерацию ответа от LLM")
        
        if not request.message:
            raise HTTPException(
                status_code=400,
                detail="Поле 'message' обязательно для заполнения"
            )
        
        response = await llm_service.generate_response(request)
        
        logger.info("Ответ успешно сгенерирован")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
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
