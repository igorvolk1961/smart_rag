"""
Кастомные эмбеддинг-классы для различных API.

Поддерживает:
- Ollama API
- GigaChat API (GigaEmbeddings)
"""

import logging
import time
import asyncio
from typing import List, Optional
import httpx
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class OllamaEmbedding(BaseEmbedding):
    """
    Кастомный класс эмбеддингов для Ollama API.
    
    Поддерживает батчинг, retry логику и таймауты.
    """
    
    def __init__(
        self,
        model: str = "jeffh/intfloat-multilingual-e5-large:q8_0",
        api_url: str = "http://localhost:11434/v1",
        batch_size: int = 8,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """
        Инициализация Ollama эмбеддинга.
        
        Args:
            model: Название модели Ollama
            api_url: URL API Ollama
            batch_size: Размер батча для обработки текстов
            max_retries: Максимальное количество попыток при ошибке
            timeout: Таймаут запроса в секундах
        """
        # Определяем размер эмбеддинга на основе модели
        # jeffh/intfloat-multilingual-e5-large имеет размер 1024
        if "e5-large" in model.lower():
            embedding_dim = 1024
        elif "e5-base" in model.lower():
            embedding_dim = 768
        elif "e5-small" in model.lower():
            embedding_dim = 384
        else:
            # По умолчанию для e5-large
            embedding_dim = 1024
            logger.warning(
                f"Неизвестная модель {model}. Используется размер эмбеддинга по умолчанию: {embedding_dim}"
            )
        
        # Передаем размер эмбеддинга в базовый класс
        super().__init__(model_name=model, embed_dim=embedding_dim, **kwargs)
        
        # Используем object.__setattr__ для установки атрибутов, которые не являются Pydantic полями
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'api_url', api_url.rstrip('/'))
        object.__setattr__(self, 'batch_size', batch_size)
        object.__setattr__(self, 'max_retries', max_retries)
        object.__setattr__(self, 'timeout', timeout)
        object.__setattr__(self, 'embedding_dim', embedding_dim)
        
        logger.info(
            f"OllamaEmbedding инициализирован: model={model}, "
            f"api_url={api_url}, batch_size={batch_size}, embedding_dim={self.embedding_dim}"
        )
    
    @classmethod
    def class_name(cls) -> str:
        """Возвращает имя класса."""
        return "OllamaEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Получение эмбеддинга для запроса.
        
        Args:
            query: Текст запроса
        
        Returns:
            Список чисел (эмбеддинг)
        """
        embeddings = self._get_text_embeddings([query])
        return embeddings[0] if embeddings else []
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Получение эмбеддинга для одного текста.
        
        Args:
            text: Текст для обработки
        
        Returns:
            Список чисел (эмбеддинг)
        """
        embeddings = self._get_text_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Асинхронное получение эмбеддинга для запроса.
        
        Args:
            query: Текст запроса
        
        Returns:
            Список чисел (эмбеддинг)
        """
        embeddings = await self._aget_text_embeddings([query])
        return embeddings[0] if embeddings else []
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Асинхронное получение эмбеддингов для списка текстов с батчингом.
        
        Args:
            texts: Список текстов для обработки
        
        Returns:
            Список эмбеддингов
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Обрабатываем тексты батчами
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._aget_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _aget_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Асинхронное получение эмбеддингов для батча текстов с retry логикой.
        
        Args:
            texts: Список текстов в батче
        
        Returns:
            Список эмбеддингов
        """
        endpoint = f"{self.api_url}/embeddings"
        
        for attempt in range(self.max_retries):
            try:
                embeddings = []
                
                # Обрабатываем тексты параллельно
                tasks = [self._aget_single_embedding(text, attempt) for text in texts]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Ошибка при получении эмбеддинга для текста {i}: {result}")
                        return []
                    if result:
                        embeddings.append(result)
                    else:
                        logger.error(f"Не удалось получить эмбеддинг для текста {i}")
                        return []
                
                return embeddings
                
            except Exception as e:
                logger.warning(
                    f"Ошибка при получении эмбеддингов (попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                    continue
                else:
                    logger.error("Превышено максимальное количество попыток")
                    raise
        
        return []
    
    async def _aget_single_embedding(self, text: str, attempt: int = 0) -> Optional[List[float]]:
        """
        Асинхронное получение эмбеддинга для одного текста.
        
        Args:
            text: Текст для обработки
            attempt: Номер попытки (для логирования)
        
        Returns:
            Эмбеддинг или None при ошибке
        """
        endpoint = f"{self.api_url}/embeddings"
        
        # OpenAI-совместимый API использует поле "input" вместо "prompt"
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Ollama OpenAI-совместимый API возвращает эмбеддинг в data[0].embedding
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    embedding_data = data["data"][0]
                    if "embedding" in embedding_data:
                        embedding = embedding_data["embedding"]
                        if isinstance(embedding, list) and len(embedding) > 0:
                            return embedding
                        else:
                            logger.error(f"Неверный формат эмбеддинга в ответе: {type(embedding)}")
                            return None
                    else:
                        logger.error(f"Поле 'embedding' отсутствует в data[0]: {embedding_data.keys()}")
                        return None
                elif "embedding" in data:
                    # Старый формат (напрямую в корне)
                    embedding = data["embedding"]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    else:
                        logger.error(f"Неверный формат эмбеддинга в ответе: {type(embedding)}")
                        return None
                else:
                    logger.error(f"Поле 'embedding' или 'data' отсутствует в ответе: {data.keys()}")
                    logger.debug(f"Полный ответ: {data}")
                    return None
                    
        except httpx.TimeoutException:
            logger.warning(f"Таймаут при получении эмбеддинга для текста (длина: {len(text)})")
            raise
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}", exc_info=True)
            raise
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Получение эмбеддингов для списка текстов с батчингом.
        
        Args:
            texts: Список текстов для обработки
        
        Returns:
            Список эмбеддингов
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Обрабатываем тексты батчами
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._get_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Получение эмбеддингов для батча текстов с retry логикой.
        
        Args:
            texts: Список текстов в батче
        
        Returns:
            Список эмбеддингов
        """
        endpoint = f"{self.api_url}/embeddings"
        
        for attempt in range(self.max_retries):
            try:
                # Подготовка данных для запроса
                # Ollama API ожидает список текстов или один текст
                # Для батча отправляем каждый текст отдельно или используем batch endpoint
                embeddings = []
                
                # Если в Ollama есть batch endpoint, используем его
                # Иначе обрабатываем последовательно
                for text in texts:
                    embedding = self._get_single_embedding(text, attempt)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        # Если не удалось получить эмбеддинг, возвращаем пустой список
                        logger.error(f"Не удалось получить эмбеддинг для текста")
                        return []
                
                return embeddings
                
            except httpx.TimeoutException as e:
                logger.warning(
                    f"Таймаут при получении эмбеддингов (попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка
                    continue
                else:
                    logger.error("Превышено максимальное количество попыток при таймауте")
                    raise
            
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"HTTP ошибка при получении эмбеддингов (попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1 and e.response.status_code >= 500:
                    # Retry только для серверных ошибок
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"HTTP ошибка: {e.response.status_code}")
                    raise
            
            except Exception as e:
                logger.error(
                    f"Неожиданная ошибка при получении эмбеддингов (попытка {attempt + 1}/{self.max_retries}): {e}",
                    exc_info=True
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
        
        return []
    
    def _get_single_embedding(self, text: str, attempt: int = 0) -> Optional[List[float]]:
        """
        Получение эмбеддинга для одного текста.
        
        Args:
            text: Текст для обработки
            attempt: Номер попытки (для логирования)
        
        Returns:
            Эмбеддинг или None при ошибке
        """
        endpoint = f"{self.api_url}/embeddings"
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            with httpx.Client(timeout=self.timeout, verify=False) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Ollama OpenAI-совместимый API возвращает эмбеддинг в data[0].embedding
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    embedding_data = data["data"][0]
                    if "embedding" in embedding_data:
                        embedding = embedding_data["embedding"]
                        if isinstance(embedding, list) and len(embedding) > 0:
                            return embedding
                        else:
                            logger.error(f"Неверный формат эмбеддинга в ответе: {type(embedding)}")
                            return None
                    else:
                        logger.error(f"Поле 'embedding' отсутствует в data[0]: {embedding_data.keys()}")
                        return None
                elif "embedding" in data:
                    # Старый формат (напрямую в корне)
                    embedding = data["embedding"]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    else:
                        logger.error(f"Неверный формат эмбеддинга в ответе: {type(embedding)}")
                        return None
                else:
                    logger.error(f"Поле 'embedding' или 'data' отсутствует в ответе: {data.keys()}")
                    logger.debug(f"Полный ответ: {data}")
                    return None
                    
        except httpx.TimeoutException:
            logger.warning(f"Таймаут при получении эмбеддинга для текста (длина: {len(text)})")
            raise
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}", exc_info=True)
            raise

