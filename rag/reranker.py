"""
Модуль для реранкинга результатов поиска.

Поддерживает реранкинг через OpenAI-совместимый API.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
import json

logger = logging.getLogger(__name__)


class ChatCompletionsReranker:
    """
    Класс для реранкинга результатов поиска через OpenAI-совместимый API.
    
    Использует модели реранкера для переупорядочивания результатов
    по релевантности к запросу.
    
    Использует OpenAI-совместимый API (/v1/chat/completions).
    Работает с любым провайдером, поддерживающим этот формат.
    """
    
    def __init__(
        self,
        model: str = "dengcao/Qwen3-Reranker-0.6B:F16",
        api_url: str = "http://localhost:11434",
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Инициализация реранкера.
        
        Args:
            model: Название модели для реранкинга
            api_url: URL API (OpenAI-совместимый endpoint)
            max_retries: Максимальное количество попыток при ошибке
            timeout: Таймаут запроса в секундах
        """
        self.model = model
        self.api_url = api_url.rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(
            f"ChatCompletionsReranker инициализирован: model={model}, "
            f"api_url={api_url}"
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Реранкинг документов по релевантности к запросу.
        
        Args:
            query: Текст запроса
            documents: Список документов для реранкинга.
                      Каждый документ должен содержать ключи:
                      - "text": текст документа
                      - "score": текущая оценка (опционально)
                      - "id": идентификатор документа
                      - "metadata": метаданные (опционально)
            top_k: Количество топ-результатов для возврата (если None, возвращает все)
        
        Returns:
            Список документов, отсортированных по релевантности
        """
        if not documents:
            logger.warning("Пустой список документов для реранкинга")
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            # Формируем промпт для реранкера
            rerank_prompt = self._create_rerank_prompt(query, documents)
            
            # Выполняем запрос к API
            scores = self._get_rerank_scores(rerank_prompt, len(documents))
            
            if not scores or len(scores) != len(documents):
                logger.warning(
                    f"Несоответствие количества оценок ({len(scores) if scores else 0}) "
                    f"и документов ({len(documents)})"
                )
                # Возвращаем исходный порядок, если реранкинг не удался
                return documents[:top_k]
            
            # Объединяем документы с оценками реранкера
            reranked_docs = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["rerank_score"] = score
                # Обновляем общий score (можно комбинировать с исходным)
                original_score = doc.get("score", 0.0)
                # Нормализуем score реранкера к диапазону 0-1 и комбинируем
                normalized_rerank_score = max(0.0, min(1.0, score))
                reranked_doc["score"] = (original_score * 0.3) + (normalized_rerank_score * 0.7)
                reranked_docs.append(reranked_doc)
            
            # Сортируем по новой оценке
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)
            
            logger.debug(
                f"Реранкинг завершен: {len(reranked_docs)} документов, "
                f"топ-{top_k} результатов"
            )
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка при реранкинге: {e}", exc_info=True)
            # В случае ошибки возвращаем исходный порядок
            return documents[:top_k]
    
    def _create_rerank_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для реранкера.
        
        Args:
            query: Текст запроса
            documents: Список документов
        
        Returns:
            Промпт для модели реранкера
        """
        # Формируем список документов для промпта
        docs_text = ""
        for idx, doc in enumerate(documents, start=1):
            text = doc.get("text", "")
            # Ограничиваем длину текста для промпта
            if len(text) > 500:
                text = text[:500] + "..."
            docs_text += f"Document {idx}:\n{text}\n\n"
        
        prompt = f"""Ты - эксперт по оценке релевантности документов к запросу.

Запрос: {query}

Документы для оценки:
{docs_text}

Оцени релевантность каждого документа к запросу по шкале от 0.0 до 1.0, где:
- 1.0 = полностью релевантен
- 0.5 = частично релевантен
- 0.0 = не релевантен

Верни только JSON массив с оценками в том же порядке, что и документы.
Формат: [0.95, 0.72, 0.45, ...]

Оценки:"""
        
        return prompt
    
    def _get_rerank_scores(self, prompt: str, expected_count: int) -> List[float]:
        """
        Получение оценок релевантности от модели реранкера.
        
        Использует OpenAI-совместимый API (/v1/chat/completions).
        
        Args:
            prompt: Промпт для модели
            expected_count: Ожидаемое количество оценок
        
        Returns:
            Список оценок релевантности
        """
        import httpx
        
        # Используем OpenAI-совместимый endpoint
        chat_url = f"{self.api_url}/v1/chat/completions"
        
        for attempt in range(self.max_retries):
            try:
                response = httpx.post(
                    chat_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.0,  # Детерминированный вывод
                        "max_tokens": 500,    # Ограничение длины ответа
                        "stream": False
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                # OpenAI-совместимый формат ответа
                if "choices" not in result or len(result["choices"]) == 0:
                    raise ValueError(
                        f"Неожиданный формат ответа от API: отсутствует поле 'choices'. "
                        f"Ответ: {result}"
                    )
                
                response_text = result["choices"][0]["message"]["content"].strip()
                
                # Парсим JSON из ответа
                scores = self._parse_scores(response_text, expected_count)
                
                if scores and len(scores) == expected_count:
                    return scores
                else:
                    logger.warning(
                        f"Попытка {attempt + 1}: Неверное количество оценок. "
                        f"Ожидалось: {expected_count}, получено: {len(scores) if scores else 0}"
                    )
                    
            except httpx.TimeoutException:
                logger.warning(f"Попытка {attempt + 1}: Таймаут при запросе к API")
            except httpx.HTTPStatusError as e:
                logger.error(f"Попытка {attempt + 1}: HTTP ошибка {e.response.status_code}: {e}")
                if e.response.status_code == 404:
                    raise RuntimeError(
                        f"OpenAI-совместимый API не найден по адресу {chat_url}. "
                        f"Убедитесь, что сервер запущен и поддерживает /v1/chat/completions endpoint."
                    ) from e
            except Exception as e:
                logger.error(f"Попытка {attempt + 1}: Ошибка при запросе к API: {e}")
            
            if attempt < self.max_retries - 1:
                import time
                time.sleep(1 * (attempt + 1))  # Экспоненциальная задержка
        
        raise RuntimeError(
            f"Не удалось получить оценки реранкера после {self.max_retries} попыток. "
            f"Проверьте подключение к API по адресу {self.api_url}"
        )
    
    def _parse_scores(self, response_text: str, expected_count: int) -> List[float]:
        """
        Парсинг оценок из ответа модели.
        
        Args:
            response_text: Текст ответа модели
            expected_count: Ожидаемое количество оценок
        
        Returns:
            Список оценок
        """
        try:
            # Пытаемся найти JSON массив в ответе
            # Модель может вернуть текст с JSON внутри
            import re
            
            # Ищем JSON массив
            json_match = re.search(r'\[[\d\s.,]+\]', response_text)
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                
                if isinstance(scores, list):
                    # Преобразуем в float и ограничиваем диапазоном 0-1
                    scores = [max(0.0, min(1.0, float(score))) for score in scores]
                    return scores
            
            # Если не нашли JSON, пытаемся парсить числа из текста
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                scores = [max(0.0, min(1.0, float(num))) for num in numbers[:expected_count]]
                # Нормализуем, если числа больше 1
                max_score = max(scores) if scores else 1.0
                if max_score > 1.0:
                    scores = [s / max_score for s in scores]
                return scores[:expected_count]
            
            logger.warning(f"Не удалось распарсить оценки из ответа: {response_text[:200]}")
            return []
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге оценок: {e}")
            return []

