"""
Утилита для очистки коллекции Qdrant.

Использование:
    python clear_collection.py                    # Показать статистику
    python clear_collection.py --clear            # Очистить коллекцию
    python clear_collection.py --recreate         # Пересоздать коллекцию
    python clear_collection.py --delete DOC_ID    # Удалить конкретный документ
    python clear_collection.py --list             # Показать список документов
"""

import argparse
import sys
from loguru import logger
from utils.logging import setup_logging
from utils.config import get_config
from utils.collection_manager import CollectionManager

# Настройка логирования
config = get_config()
logging_config = config.get("logging", {})
setup_logging(
    level=logging_config.get("level", "INFO"),
    log_format=logging_config.get("format"),
    clear_log_on_start=logging_config.get("clear_log_on_start", True)
)


def main():
    """Основная функция утилиты."""
    parser = argparse.ArgumentParser(
        description="Утилита для управления коллекцией Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python clear_collection.py                    # Показать статистику коллекции
  python clear_collection.py --clear            # Очистить коллекцию (удалить все точки)
  python clear_collection.py --recreate         # Пересоздать коллекцию
  python clear_collection.py --delete doc123    # Удалить документ с ID 'doc123'
  python clear_collection.py --list             # Показать список всех документов
        """
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Очистить коллекцию (удалить все точки)"
    )
    
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Пересоздать коллекцию (удалить и создать заново)"
    )
    
    parser.add_argument(
        "--delete",
        type=str,
        metavar="DOC_ID",
        help="Удалить конкретный документ по ID"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список всех документов в коллекции"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Показать статистику коллекции (по умолчанию)"
    )
    
    args = parser.parse_args()
    
    # Если не указаны аргументы, показываем статистику
    if not any([args.clear, args.recreate, args.delete, args.list, args.stats]):
        args.stats = True
    
    try:
        manager = CollectionManager()
        
        if args.stats:
            logger.info("Получение статистики коллекции...")
            stats = manager.get_collection_stats()
            
            if "error" in stats:
                logger.error(f"Ошибка: {stats['error']}")
                return 1
            
            logger.info("=" * 60)
            logger.info("СТАТИСТИКА КОЛЛЕКЦИИ")
            logger.info("=" * 60)
            logger.info(f"Название: {stats['collection_name']}")
            logger.info(f"Всего точек: {stats['total_points']}")
            logger.info(f"Всего документов: {stats['total_documents']}")
            logger.info(f"Всего чанков: {stats['total_chunks']}")
            logger.info(f"Статус: {stats.get('status', 'N/A')}")
            
            if stats.get('config'):
                logger.info(f"Размер вектора: {stats['config'].get('vector_size', 'N/A')}")
                logger.info(f"Метрика расстояния: {stats['config'].get('distance', 'N/A')}")
            
            if stats.get('documents'):
                logger.info("\nДокументы в коллекции:")
                for doc in stats['documents']:
                    logger.info(f"  - {doc['document_id']}: {doc['chunks_count']} чанков")
            
            logger.info("=" * 60)
        
        elif args.list:
            logger.info("Получение списка документов...")
            documents = manager.list_documents()
            
            if not documents:
                logger.info("Коллекция пуста")
            else:
                logger.info(f"\nНайдено документов: {len(documents)}")
                logger.info("-" * 60)
                for doc in documents:
                    logger.info(f"ID: {doc['document_id']}")
                    logger.info(f"  Путь: {doc.get('document_path', 'N/A')}")
                    logger.info(f"  Чанков: {doc['chunks_count']}")
                    logger.info("-" * 60)
        
        elif args.clear:
            logger.warning("⚠ ВНИМАНИЕ: Будет удалены ВСЕ точки из коллекции!")
            response = input("Продолжить? (yes/no): ")
            
            if response.lower() in ['yes', 'y', 'да', 'д']:
                result = manager.clear_collection()
                
                if result.get("success"):
                    logger.info(f"✅ Коллекция очищена успешно")
                    logger.info(f"   Удалено точек: {result['points_deleted']}")
                else:
                    logger.error(f"❌ Ошибка при очистке: {result.get('error')}")
                    return 1
            else:
                logger.info("Операция отменена")
        
        elif args.recreate:
            logger.warning("⚠ ВНИМАНИЕ: Коллекция будет удалена и создана заново!")
            response = input("Продолжить? (yes/no): ")
            
            if response.lower() in ['yes', 'y', 'да', 'д']:
                result = manager.recreate_collection()
                
                if result.get("success"):
                    logger.info(f"✅ Коллекция успешно пересоздана")
                else:
                    logger.error(f"❌ Ошибка при пересоздании: {result.get('error')}")
                    return 1
            else:
                logger.info("Операция отменена")
        
        elif args.delete:
            logger.info(f"Удаление документа: {args.delete}")
            result = manager.delete_document(args.delete)
            
            if result.get("success"):
                logger.info(f"✅ Документ удален успешно")
                logger.info(f"   Удалено точек: {result['points_deleted']}")
            else:
                logger.error(f"❌ Ошибка при удалении: {result.get('error', 'Неизвестная ошибка')}")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nОперация прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

