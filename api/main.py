"""
Главное FastAPI приложение.
"""

import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from api.routes import llm_routes
from api.models.llm_models import error_response_body
from utils.logging import setup_logging
from utils.config import get_config


def create_app() -> FastAPI:
    """
    Создание и настройка FastAPI приложения.
    
    Returns:
        Настроенное FastAPI приложение
    """
    # Загружаем конфигурацию
    config = get_config()
    
    # Настраиваем логирование
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    
    logger.info("Инициализация FastAPI приложения")
    
    # Создаем приложение
    app = FastAPI(
        title="Smart RAG API",
        description="API для работы с RAG-системой и LLM",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Настраиваем CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В продакшене нужно указать конкретные домены
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Перехват любых 4xx/5xx (в т.ч. 400 от FastAPI): отдаём 200 с телом { error, detail, code }
    @app.middleware("http")
    async def errors_to_200(request: Request, call_next):
        response = await call_next(request)
        if response.status_code < 400:
            return response
        body_bytes = b""
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                body_bytes += chunk
        elif getattr(response, "body", None) is not None:
            body_bytes = response.body if isinstance(response.body, bytes) else response.body.encode()
        detail = "Ошибка запроса"
        code = "error"
        errors = None
        if body_bytes:
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                code = data.get("code", "error")
                if isinstance(data.get("detail"), str):
                    detail = data["detail"]
                elif isinstance(data.get("detail"), list):
                    parts = [e.get("msg", str(e)) for e in data["detail"] if isinstance(e, dict)]
                    if parts:
                        detail = "; ".join(parts)
                if isinstance(data.get("errors"), list):
                    errors = data["errors"]
            except Exception:
                try:
                    detail = body_bytes.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
        return JSONResponse(
            status_code=200,
            content=error_response_body(
                error="Ошибка запроса",
                detail=detail,
                code=code,
                errors=errors,
            ),
        )

    # Единый формат ошибок: {"error", "detail", "code"}
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response_body(
                error="Ошибка запроса",
                detail=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        raw = exc.errors()
        errors_list = []
        for e in raw:
            loc = e.get("loc", ())
            # Убираем "body" в начале для краткости, путь: current_message или chat_history.0.role
            path_parts = [str(x) for x in loc if x != "body"]
            field = ".".join(path_parts) if path_parts else (loc[0] if loc else "?")
            errors_list.append({
                "field": field,
                "message": e.get("msg", "Ошибка валидации"),
                "type": e.get("type"),
            })
        # Краткое общее описание
        detail_parts = [f"{item['field']}: {item['message']}" for item in errors_list]
        detail_summary = "; ".join(detail_parts) if detail_parts else "Неверный формат или состав полей запроса."
        return JSONResponse(
            status_code=400,
            content=error_response_body(
                error="Ошибка валидации запроса",
                detail=detail_summary,
                code="validation_error",
                errors=errors_list,
            ),
        )

    # Подключаем роуты
    app.include_router(llm_routes.router)

    @app.get("/", tags=["Root"])
    async def root():
        """Корневой эндпоинт."""
        return {
            "message": "Smart RAG API",
            "version": "0.1.0",
            "docs": "/docs"
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Проверка здоровья приложения."""
        return {"status": "ok"}
    
    logger.info("FastAPI приложение успешно инициализировано")
    
    return app


# Создаем экземпляр приложения
app = create_app()
