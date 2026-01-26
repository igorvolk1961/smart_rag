"""
Главное FastAPI приложение.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import llm_routes
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
    
    @app.post("/health", tags=["Health"])
    async def health_check():
        """Проверка здоровья приложения."""
        return {"status": "ok"}
    
    logger.info("FastAPI приложение успешно инициализировано")
    
    return app


# Создаем экземпляр приложения
app = create_app()
