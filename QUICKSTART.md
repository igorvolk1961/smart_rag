# Быстрый старт Smart RAG

## Шаг 1: Проверка зависимостей

Перед запуском убедитесь, что установлены и запущены:

### 1.1 Python окружение

```bash
# Проверка версии Python (требуется 3.8+)
python --version

# Создание виртуального окружения (если еще не создано)
python -m venv .venv

# Активация виртуального окружения
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 1.2 Установка зависимостей Python

```bash
pip install -r requirements.txt
```

### 1.3 Qdrant

Qdrant должен быть запущен локально.

**Проверка:**
- Откройте браузер и перейдите на `http://localhost:6333/` (должен показать JSON с информацией о версии)
- Или проверьте через командную строку:
  ```bash
  curl http://localhost:6333/
  ```
- Проверка списка коллекций:
  ```bash
  curl http://localhost:6333/collections
  ```
  
**Примечание:** В некоторых версиях Qdrant (например, 1.15.5) endpoint `/dashboard` может быть недоступен. Используйте корневой endpoint `/` для проверки.

**Если Qdrant не запущен:**
- Запустите Qdrant согласно вашей конфигурации:
  ```bash
  f:/programs/qdrant/qdrant.exe --config-path f:/programs/qdrant/qdrant_config.yaml
  ```

### 1.4 Ollama

Ollama должен быть запущен с установленной моделью.

**Проверка:**
```bash
# Проверка, что Ollama запущен
curl http://localhost:11434/api/tags

# Проверка наличия модели
ollama list
```

**Если модель не установлена:**
```bash
# Установка модели (может занять время)
ollama pull jeffh/intfloat-multilingual-e5-large:q8_0
```

**Если Ollama не запущен:**
- Запустите Ollama сервис
- Или запустите через командную строку: `ollama serve`

## Шаг 2: Настройка конфигурации

### 2.1 Проверка config.yaml

Убедитесь, что пути в `config.yaml` корректны:
- `qdrant.executable_path` - путь к Qdrant
- `qdrant.config_path` - путь к конфигу Qdrant
- `qdrant.url` - URL Qdrant API (по умолчанию `http://localhost:6333`)
- `embeddings.api_url` - URL Ollama API (по умолчанию `http://localhost:11434/v1`)

### 2.2 Создание .env файла (опционально)

```bash
# Скопируйте пример
copy .env.example .env

# Отредактируйте .env при необходимости
```

## Шаг 3: Запуск программы

### Вариант 1: Запуск примера использования

```bash
python example_usage.py
```

Этот скрипт:
- Проверяет подключение к Qdrant и Ollama
- Показывает информацию о коллекции
- Содержит закомментированные примеры для индексации и поиска

### Вариант 2: Интерактивное использование через Python

```python
from rag import RAGPipeline

# Инициализация
rag = RAGPipeline()

# Индексация документа
result = rag.index_document("путь/к/документу.docx")
print(f"Проиндексировано: {result['nodes_indexed']} узлов")

# Поиск
results = rag.search("ваш запрос", top_k=5)
for r in results:
    print(f"Score: {r['score']:.4f}, Text: {r['text'][:100]}...")

# Получение контекста
context = rag.get_context("ваш запрос", top_k=5)
print(context)
```

### Вариант 3: Запуск через скрипт проверки

```bash
python check_system.py
```

Этот скрипт проверит все зависимости и готовность системы.

## Шаг 4: Тестирование

### 4.1 Проверка подключений

```python
from rag import RAGPipeline

rag = RAGPipeline()

# Проверка информации о коллекции
info = rag.get_collection_info()
print(f"Коллекция: {info['name']}")
print(f"Точек в коллекции: {info['vectors_count']}")
```

### 4.2 Индексация тестового документа

```python
# Создайте папку data/input и поместите туда тестовый документ
rag.index_document("data/input/test.docx")
```

### 4.3 Поиск

```python
results = rag.search("ваш вопрос", top_k=5)
print(f"Найдено {len(results)} результатов")
```

## Возможные проблемы

### Ошибка подключения к Qdrant

```
ConnectionError: Could not connect to Qdrant
```

**Решение:**
- Убедитесь, что Qdrant запущен
- Проверьте URL в `config.yaml` (по умолчанию `http://localhost:6333`)
- Проверьте, что порт не занят другим процессом

### Ошибка подключения к Ollama

```
httpx.ConnectError: Connection refused
```

**Решение:**
- Убедитесь, что Ollama запущен: `ollama serve`
- Проверьте URL в `config.yaml` (по умолчанию `http://localhost:11434/v1`)
- Проверьте, что модель установлена: `ollama list`

### Ошибка импорта SmartChanker

```
ImportError: SmartChanker не установлен
```

**Решение:**
```bash
pip install git+https://github.com/igorvolk1961/smart_chanker.git
```

### Ошибка при индексации документа

```
FileNotFoundError: Документ не найден
```

**Решение:**
- Проверьте путь к документу
- Убедитесь, что файл существует
- Поддерживаемые форматы: `.docx`, `.txt`, `.pdf`

## Следующие шаги

После успешного запуска:
1. Индексируйте ваши документы
2. Протестируйте поиск
3. Переходите к Этапу 3: Использование метаданных для улучшения контекста

