# Smart RAG

RAG-система для работы с документами, использующая интеллектуальный чанкер [SmartChanker](https://github.com/igorvolk1961/smart_chanker) для создания структурированных чанков с сохранением иерархической структуры документов.

## Технологический стек

- **Чанкер**: SmartChanker
- **RAG Framework**: LlamaIndex
- **Векторное хранилище**: Qdrant
- **Эмбеддинги**: Ollama API (jeffh/intfloat-multilingual-e5-large:q8_0)
- **Backend**: FastAPI
- **Desktop App**: Для отладки и тестирования

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd smart_rag
```

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройте конфигурацию:
   - Скопируйте `.env.example` в `.env` и настройте параметры
   - Проверьте `config.yaml` и при необходимости измените пути

5. Убедитесь, что запущены необходимые сервисы:
   - Qdrant (локально или удаленно)
   - Ollama с моделью `jeffh/intfloat-multilingual-e5-large:q8_0`

## Структура проекта

```
smart_rag/
├── rag/                    # Основной RAG модуль
│   ├── chunker_integration.py
│   └── ...
├── api/                    # FastAPI приложение
│   ├── routes/
│   ├── models/
│   └── ...
├── desktop/               # Десктоп-приложение
│   └── ui/
├── utils/                 # Утилиты
│   ├── config.py
│   └── logging.py
├── config.yaml            # Конфигурация RAG-системы
├── .env.example           # Пример переменных окружения
├── requirements.txt       # Зависимости
└── DEVELOPMENT_PLAN.md    # План разработки
```

## Использование

### Запуск API сервера

```bash
python -m api.main
```

или

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Запуск десктоп-приложения

```bash
python -m desktop.main
```

## Конфигурация

Основные параметры настраиваются в `config.yaml`:

- **qdrant**: Настройки векторного хранилища
- **embeddings**: Настройки эмбеддингов (Ollama)
- **chunker**: Настройки SmartChanker
- **rag**: Параметры RAG-системы
- **api**: Настройки FastAPI сервера

Дополнительные параметры можно задать через переменные окружения в `.env` файле.

## Разработка

См. [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) для подробного плана разработки.

## Лицензия

[Укажите лицензию проекта]

