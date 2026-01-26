"""
Утилиты для загрузки и работы с конфигурацией.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Logging
    log_level: str = "DEBUG"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Игнорировать дополнительные переменные окружения
    }


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
    
    Returns:
        Словарь с конфигурацией
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_env_file(env_path: str = ".env") -> None:
    """
    Загрузка переменных окружения из .env файла.
    
    Args:
        env_path: Путь к .env файлу
    """
    env_file = Path(env_path)
    
    if env_file.exists():
        load_dotenv(env_path)
    else:
        # Создаем .env из .env.example если он существует
        env_example = Path(".env.example")
        if env_example.exists():
            print(f"Внимание: файл {env_path} не найден. Используйте .env.example как шаблон.")


def get_config() -> Dict[str, Any]:
    """
    Получение полной конфигурации (YAML + переменные окружения).
    
    Returns:
        Объединенная конфигурация
    """
    # Загружаем переменные окружения
    load_env_file()
    
    # Загружаем YAML конфигурацию
    config = load_config()
    
    # Переопределяем значения из переменных окружения
    settings = Settings()
    
    # Обновляем конфигурацию значениями из окружения
    if settings.qdrant_url != "http://localhost:6333":
        config.setdefault("qdrant", {})["url"] = settings.qdrant_url
    
    if settings.api_host != "0.0.0.0":
        config.setdefault("api", {})["host"] = settings.api_host
    
    if settings.api_port != 8000:
        config.setdefault("api", {})["port"] = settings.api_port
    
    if settings.debug:
        config.setdefault("api", {})["debug"] = settings.debug
    
    if settings.log_level != "INFO":
        config.setdefault("logging", {})["level"] = settings.log_level
    
    return config

