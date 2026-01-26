"""
Точка входа для запуска FastAPI приложения.
"""

import uvicorn
from utils.config import get_config
from utils.logging import setup_logging


def main():
    """Запуск FastAPI приложения."""
    # Загружаем конфигурацию
    config = get_config()
    
    # Настраиваем логирование
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    
    # Получаем параметры API из конфигурации
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    debug = api_config.get("debug", False)
    
    # Запускаем сервер
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level.lower()
    )


if __name__ == "__main__":
    main()
