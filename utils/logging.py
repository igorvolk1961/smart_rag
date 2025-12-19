"""
Настройка логирования для RAG-системы.
"""

import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    clear_log_on_start: bool = True
) -> None:
    """
    Настройка логирования с использованием loguru.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат логов (если None, используется стандартный)
        log_file: Путь к файлу для сохранения логов (если None, только консоль)
        clear_log_on_start: Очищать лог-файл при старте приложения (по умолчанию True)
    """
    # Удаляем стандартный обработчик
    logger.remove()
    
    # Перехватываем все сообщения стандартного logging и перенаправляем в loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Получаем соответствующий уровень loguru
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Находием фрейм, откуда был вызван логгер
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Устанавливаем перехватчик для всех стандартных логгеров
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Формат по умолчанию
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Добавляем обработчик для консоли
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Добавляем обработчик для файла (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Очистка лог-файла при старте, если указано
        if clear_log_on_start and log_path.exists():
            log_path.unlink()
        
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    logger.info(f"Логирование настроено. Уровень: {level}")


def get_logger(name: str):
    """
    Получение логгера с указанным именем.
    
    Args:
        name: Имя логгера (обычно __name__)
    
    Returns:
        Логгер loguru
    """
    return logger.bind(name=name)

