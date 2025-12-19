# Инструкция по отправке проекта в GitHub

## Шаг 1: Инициализация Git репозитория (если еще не инициализирован)

```bash
git init
```

## Шаг 2: Добавление удаленного репозитория

```bash
git remote add origin https://github.com/igorvolk1961/smart_rag.git
```

Если репозиторий уже существует, проверьте текущие remote:
```bash
git remote -v
```

Если нужно изменить URL:
```bash
git remote set-url origin https://github.com/igorvolk1961/smart_rag.git
```

## Шаг 3: Добавление файлов

```bash
git add .
```

Или выборочно:
```bash
git add README.md
git add requirements.txt
git add config.yaml
git add config.json
git add .gitignore
git add rag/
git add api/
git add desktop/
git add utils/
git add *.py
git add *.md
```

## Шаг 4: Создание первого коммита

```bash
git commit -m "Initial commit: Smart RAG система с интеграцией SmartChanker, LlamaIndex и Qdrant"
```

## Шаг 5: Отправка в GitHub

```bash
git branch -M main
git push -u origin main
```

Если возникнут проблемы с аутентификацией, используйте Personal Access Token вместо пароля.

## Проверка статуса

После отправки проверьте:
```bash
git status
git log --oneline
```

## Важные замечания

1. **Файл `.env` не будет отправлен** - он в `.gitignore`
2. **Папка `data/chunks/` не будет отправлена** - она в `.gitignore`
3. **Виртуальное окружение `.venv/` не будет отправлено** - оно в `.gitignore`

## Структура файлов для отправки

Следующие файлы и папки будут отправлены:
- `rag/` - основной RAG модуль
- `api/` - FastAPI приложение
- `desktop/` - десктоп-приложение
- `utils/` - утилиты
- `config.yaml` - конфигурация
- `config.json` - конфигурация SmartChanker
- `requirements.txt` - зависимости
- `README.md` - документация
- `QUICKSTART.md` - быстрый старт
- `DEVELOPMENT_PLAN.md` - план разработки
- `.gitignore` - игнорируемые файлы
- `.env.example` - пример переменных окружения
- `example_usage.py` - пример использования
- `test_indexing.py` - тестовый скрипт
- `check_system.py` - скрипт проверки системы

