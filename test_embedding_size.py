"""
Тест для проверки максимального размера фрагмента, который может обработать эмбеддер.

Использование:
1. Подставьте нужный текст фрагмента в переменную TEST_TEXT
2. Запустите скрипт: python test_embedding_size.py
3. Скрипт покажет результат и размеры payload
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.giga_embeddings import GigaEmbedding
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO")

# ============================================
# ВСТАВЬТЕ СЮДА ТЕКСТ ДЛЯ ТЕСТИРОВАНИЯ
# ============================================
TEST_TEXT = """6.3.1.2. СЕЗОННОЕ ПЛАНИРОВАНИЕ ПОСТАВОК ДЛЯ ДВУХ БЕРЕГОВ
6.3.1.3. ОРГАНИЗАЦИЯ ПОГРУЗОЧНО-РАЗГРУЗОЧНЫХ РАБОТ НА ОБОИХ БЕРЕГАХ
6.3.1.4. УПРАВЛЕНИЕ ГРУЗОПОТОКАМИ И ПРОПУСКНОЙ СПОСОБНОСТЬЮ
6.3.1.5. МЕЖБЕРЕГОВАЯ ДОСТАВКА МАТЕРИАЛОВ И ОБОРУДОВАНИЯ
ТАБЛИЦА 6.3.1: Годовой план перевозок основных материалов с распределением по берегам (ориентировочно, на пиковый 2026 год)
6.3.2. ТРАНСПОРТ ДЛЯ ПЕРСОНАЛА
6.3.2.1. ОРГАНИЗАЦИЯ ВАХТОВОГО ТРАНСПОРТА ДЛЯ ДВУХ ПОСЕЛКОВ
6.3.2.2. РАСПИСАНИЕ И МАРШРУТЫ ДВИЖЕНИЯ С УЧЕТОМ ДВУХ БЕРЕГОВ
6.3.2.3. МЕЖБЕРЕГОВАЯ ПЕРЕПРАВА ПЕРСОНАЛА
6.3.2.4. ТРАНСПОРТ ДЛЯ ЭКСКУРСИЙ И КУЛЬТУРНЫХ МЕРОПРИЯТИЙ
6.3.2.5. СИСТЕМА БРОНИРОВАНИЯ И УЧЕТА ДЛЯ ДВУХБЕРЕГОВОЙ СИСТЕМЫ
6.3.3. СЛУЖЕБНЫЙ ТРАНСПОРТ
6.3.3.1. ОБЕСПЕЧЕНИЕ ТЕХНИЧЕСКОГО ПЕРСОНАЛА НА ОБОИХ БЕРЕГАХ
6.3.3.2. ТРАНСПОРТ ДЛЯ ОПЕРАТИВНЫХ ВЫЕЗДОВ МЕЖДУ БЕРЕГАМИ
6.3.3.3. СПЕЦИАЛЬНЫЙ ТРАНСПОРТ ДЛЯ РУКОВОДСТВА И КООРДИНАЦИИ
6.3.3.4. СИСТЕМА ДИСПЕТЧЕРИЗАЦИИ И КОНТРОЛЯ ДВУХБЕРЕГОВОЙ ЛОГИСТИКИ
6.3.4. ОРГАНИЗАЦИЯ МЕЖБЕРЕГОВОЙ СВЯЗИ И КООРДИНАЦИИ
6.3.4.1. ИНФРАСТРУКТУРА МЕЖБЕРЕГОВОЙ ПЕРЕПРАВЫ
6.3.4.2. РЕЖИМ РАБОТЫ МЕЖБЕРЕГОВОЙ ПЕРЕПРАВЫ
6.3.4.3. БЕЗОПАСНОСТЬ МЕЖБЕРЕГОВЫХ ПЕРЕВОЗОК
6.3.4.4. КООРДИНАЦИЯ ЛОГИСТИКИ МЕЖДУ ДВУМЯ БЕРЕГАМИ"""


def test_embedding_size():
    """Тест размера фрагмента для эмбеддера."""
    
    # Получаем API ключ из переменной окружения
    api_key = "M2RjNGFkZGEtOTA0MS00MzI0LTlmNzUtNzczNTIxNmQ0Zjk1OmNiMDcxYTkyLTE5MTctNDk2MS1hOWZjLTIwMjgxZDU1NWUxZg=="
#    os.getenv("GIGACHAT_AUTH_KEY")
    if not api_key:
        print("ОШИБКА: Переменная окружения GIGACHAT_AUTH_KEY не установлена")
        print("Установите её в файле .env или через переменные окружения")
        return
    
    # Инициализируем эмбеддер
    print("Инициализация GigaEmbedding...")
    try:
        embedding = GigaEmbedding(
            credentials=api_key,
            scope="GIGACHAT_API_PERS",
            api_url="https://gigachat.devices.sberbank.ru/api/v1",
            model="Embeddings",
            batch_size=1,  # Используем batch_size=1 для тестирования одного фрагмента
            max_retries=1,
            timeout=60
        )
        print("✓ GigaEmbedding инициализирован успешно")
    except Exception as e:
        print(f"✗ Ошибка инициализации: {e}")
        return
    
    # Вычисляем размеры
    import json
    text_length = len(TEST_TEXT)
    text_bytes = len(TEST_TEXT.encode('utf-8'))
    
    payload = {
        "model": embedding.model,
        "input": TEST_TEXT
    }
    json_payload_str = json.dumps(payload, ensure_ascii=False)
    json_payload_size = len(json_payload_str.encode('utf-8'))
    
    print("\n" + "="*60)
    print("ПАРАМЕТРЫ ТЕСТА:")
    print("="*60)
    print(f"Длина текста: {text_length} символов")
    print(f"Размер текста: {text_bytes} байт")
    print(f"Размер JSON payload: {json_payload_size} байт")
    print(f"batch_size: {embedding.batch_size}")
    print("="*60 + "\n")
    
    # Вызываем эмбеддер
    print("Отправка запроса к эмбеддеру...")
    try:
        result = embedding._get_single_embedding(TEST_TEXT)
        
        if result:
            print("\n" + "="*60)
            print("✓ УСПЕХ!")
            print("="*60)
            print(f"Получен эмбеддинг размером: {len(result)} элементов")
            print(f"\nМаксимальные успешные размеры после запроса:")
            print(f"  max_json_payload_size: {embedding.max_json_payload_size} байт")
            print(f"  best_text_length: {embedding.best_text_length} символов")
            print("="*60)
        else:
            print("\n✗ ОШИБКА: Эмбеддер вернул None")
            
    except RuntimeError as e:
        print("\n" + "="*60)
        print("✗ ОШИБКА 413 (Request Entity Too Large)")
        print("="*60)
        print(f"Сообщение: {e}")
        print(f"\nМаксимальные успешные размеры:")
        print(f"  max_json_payload_size: {embedding.max_json_payload_size} байт")
        print(f"  best_text_length: {embedding.best_text_length} символов")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ОШИБКА")
        print("="*60)
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {e}")
        print("="*60)


if __name__ == "__main__":
    test_embedding_size()
