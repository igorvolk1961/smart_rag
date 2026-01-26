"""
Тестовый скрипт для проверки индексации документа.
"""

from loguru import logger
from utils.logging import setup_logging
from utils.config import get_config
from rag import RAGPipeline

# Настройка логирования
config = get_config()
logging_config = config.get("logging", {})
setup_logging(
    level=logging_config.get("level", "INFO"),
    log_format=logging_config.get("format"),
    clear_log_on_start=logging_config.get("clear_log_on_start", True)
)


def main():
    """Тестирование индексации."""
    logger.info("Тест индексации документа")
    
    # Инициализация RAG-пайплайна
    try:
        rag = RAGPipeline()
        logger.info("RAG-пайплайн инициализирован")
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {e}", exc_info=True)
        return
    
    # Проверка состояния коллекции до индексации
    try:
        info_before = rag.get_collection_info()
        logger.info(f"Состояние коллекции ДО индексации: {info_before}")
        vectors_before = info_before.get('vectors_count', 0)
    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекции: {e}")
        return
    
    # Индексация документа
#    document_path = r"F:\git\irs\smart_chanker\data\План строительства моста через реку Лена.docx"
    document_path = r"F:\git\irs\smart_chanker\data\План график.docx"
    
    logger.info(f"Начало индексации документа: {document_path}")
    try:
        result = rag.index_document(document_path)
        logger.info(f"✅ Документ успешно проиндексирован!")
        logger.info(f"Результат индексации: {result}")
    except Exception as e:
        logger.error(f"❌ Ошибка при индексации документа: {e}", exc_info=True)
        return
    
    # Проверка состояния коллекции после индексации
    try:
        info_after = rag.get_collection_info()
        logger.info(f"Состояние коллекции ПОСЛЕ индексации: {info_after}")
        vectors_after = info_after.get('vectors_count', 0)
        
        logger.info(f"Добавлено векторов: {vectors_after - vectors_before}")
    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекции: {e}")


if __name__ == "__main__":
    main()

