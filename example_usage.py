"""
Пример использования RAG-системы.

Этот скрипт демонстрирует базовое использование RAG-пайплайна:
1. Индексация документа
2. Поиск по запросу
3. Получение контекста
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
    """Основная функция примера."""
    logger.info("Запуск примера использования RAG-системы")
    
    # Инициализация RAG-пайплайна
    try:
        rag = RAGPipeline()
        logger.info("RAG-пайплайн успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка при инициализации RAG-пайплайна: {e}", exc_info=True)
        return
    
    # Получение информации о коллекции
    try:
        collection_info = rag.get_collection_info()
        logger.info(f"Информация о коллекции: {collection_info}")
    except Exception as e:
        logger.warning(f"Не удалось получить информацию о коллекции: {e}")
    
    query = "чем кормят работников"
    try:
        results = rag.search(query, top_k=5)
        logger.info(f"Найдено результатов: {len(results)}")
        for i, result in enumerate(results, 1):
            logger.info(f"Результат {i}: score={result['score']:.4f}, text={result['text'][:1000]}...")
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}", exc_info=True)
    
    # Пример 3: Получение контекста
    # query = "чем кормят работников"
    # try:
    #     context = rag.get_context(query, top_k=5)
    #     logger.info(f"Контекст (длина: {len(context)} символов):\n{context[:1000]}...")
    # except Exception as e:
    #     logger.error(f"Ошибка при получении контекста: {e}", exc_info=True)
    
    logger.info("Пример завершен. Раскомментируйте нужные секции для тестирования.")


if __name__ == "__main__":
    main()

