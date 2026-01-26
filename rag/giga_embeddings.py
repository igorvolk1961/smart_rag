"""
Кастомный эмбеддинг-класс для GigaChat API (GigaEmbeddings).
"""

import logging
import time
import asyncio
from typing import List, Optional
import httpx
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class GigaEmbedding(BaseEmbedding):
    """
    Кастомный класс эмбеддингов для GigaChat API (GigaEmbeddings).
    
    Поддерживает батчинг, retry логику и таймауты.
    GigaEmbeddings - модель от Сбера для работы с русским языком.
    Размер эмбеддинга: 1024 (по умолчанию для GigaEmbeddings).
    """
    
    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
        api_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
        model: str = "Embeddings",
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """
        Инициализация GigaChat эмбеддинга.
        
        Args:
            credentials: Путь к файлу с учетными данными или токен доступа
            scope: Область доступа (GIGACHAT_API_PERS для персонального API)
            api_url: URL API GigaChat
            model: Название модели (обычно "Embeddings" для эмбеддингов)
            batch_size: Размер батча для обработки текстов
            max_retries: Максимальное количество попыток при ошибке
            timeout: Таймаут запроса в секундах
        """
        # GigaEmbeddings имеет размер 1024
        embedding_dim = 1024
        
        # Передаем размер эмбеддинга в базовый класс
        super().__init__(model_name=model, embed_dim=embedding_dim, **kwargs)
        
        # Используем object.__setattr__ для установки атрибутов
        object.__setattr__(self, 'credentials', credentials)
        object.__setattr__(self, 'scope', scope)
        object.__setattr__(self, 'api_url', api_url.rstrip('/'))
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'batch_size', batch_size)
        object.__setattr__(self, 'max_retries', max_retries)
        object.__setattr__(self, 'timeout', timeout)
        object.__setattr__(self, 'embedding_dim', embedding_dim)
        object.__setattr__(self, 'access_token', None)
        
        logger.info(
            f"GigaEmbedding инициализирован: model={model}, "
            f"api_url={api_url}, batch_size={batch_size}, embedding_dim={self.embedding_dim}"
        )
        
        # Получаем токен доступа при инициализации
        self._get_access_token()
    
    def _get_access_token(self) -> Optional[str]:
        """
        Получение токена доступа для GigaChat API.
        
        Returns:
            Токен доступа или None при ошибке
        """
        try:
            # Если токен уже есть и не истек, используем его
            if self.access_token:
                return self.access_token
            
            # Если credentials - это base64 строка (Authorization Key), используем прямой запрос
            if self.credentials and not self.credentials.endswith('.json'):
                # Это base64 ключ, получаем токен напрямую через OAuth2
                token = self._get_token_from_key(self.credentials)
                if token:
                    object.__setattr__(self, 'access_token', token)
                    logger.info("Токен доступа GigaChat получен из Authorization Key")
                    return token
            
            # Импортируем gigachat только при необходимости
            try:
                from gigachat import GigaChat
            except ImportError:
                logger.error(
                    "Библиотека gigachat не установлена. "
                    "Установите её: pip install gigachat"
                )
                return None
            
            # Создаем клиент GigaChat
            if self.credentials and self.credentials.endswith('.json'):
                # Если указан путь к файлу с учетными данными
                client = GigaChat(
                    credentials=self.credentials,
                    scope=self.scope,
                    verify_ssl_certs=False
                )
            else:
                # Используем переменные окружения или токен из конфига
                client = GigaChat(
                    scope=self.scope,
                    verify_ssl_certs=False
                )
            
            # Получаем токен
            token = client._get_access_token()
            object.__setattr__(self, 'access_token', token)
            
            logger.info("Токен доступа GigaChat получен успешно")
            return token
            
        except Exception as e:
            logger.error(f"Ошибка при получении токена доступа GigaChat: {e}", exc_info=True)
            return None
    
    def _get_token_from_key(self, auth_key: str) -> Optional[str]:
        """
        Получение токена доступа напрямую из Authorization Key через OAuth2.
        
        Args:
            auth_key: Base64-encoded Authorization Key (client_id:client_secret)
        
        Returns:
            Токен доступа или None при ошибке
        """
        try:
            import base64
            
            # URL для получения токена
            token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
            
            # Декодируем ключ для получения client_id и client_secret
            try:
                decoded = base64.b64decode(auth_key).decode('utf-8')
                client_id, client_secret = decoded.split(':', 1)
            except Exception as e:
                logger.error(f"Ошибка декодирования Authorization Key: {e}")
                return None
            
            # Подготавливаем данные для запроса
            auth_data = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            
            headers = {
                "Authorization": f"Basic {auth_data}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            
            data = {
                "scope": self.scope
            }
            
            # Отправляем запрос на получение токена
            with httpx.Client(timeout=self.timeout, verify=False) as client:
                response = client.post(token_url, headers=headers, data=data)
                response.raise_for_status()
                
                token_data = response.json()
                
                if "access_token" in token_data:
                    return token_data["access_token"]
                else:
                    logger.error(f"Токен не найден в ответе: {token_data.keys()}")
                    return None
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка при получении токена: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении токена из ключа: {e}", exc_info=True)
            return None
    
    @classmethod
    def class_name(cls) -> str:
        """Возвращает имя класса."""
        return "GigaEmbedding"
    
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
        Асинхронное получение эмбеддинга для одного текста через GigaChat API.
        
        Args:
            text: Текст для обработки
            attempt: Номер попытки (для логирования)
        
        Returns:
            Эмбеддинг или None при ошибке
        """
        endpoint = f"{self.api_url}/embeddings"
        
        # Получаем токен доступа
        token = self._get_access_token()
        if not token:
            logger.error("Не удалось получить токен доступа GigaChat")
            return None
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # GigaChat API возвращает эмбеддинг в data[0].embedding
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
                    # Альтернативный формат
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
            # Если токен истек, пытаемся обновить его
            if e.response.status_code == 401:
                logger.warning("Токен доступа истек, обновляем...")
                object.__setattr__(self, 'access_token', None)
                token = self._get_access_token()
                if token and attempt < self.max_retries - 1:
                    # Повторяем запрос с новым токеном
                    return await self._aget_single_embedding(text, attempt + 1)
            
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
        for attempt in range(self.max_retries):
            try:
                embeddings = []
                
                # Обрабатываем тексты последовательно (GigaChat может иметь ограничения)
                for text in texts:
                    embedding = self._get_single_embedding(text, attempt)
                    if embedding:
                        embeddings.append(embedding)
                    else:
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
                # Если токен истек, пытаемся обновить его
                if e.response.status_code == 401:
                    logger.warning("Токен доступа истек, обновляем...")
                    object.__setattr__(self, 'access_token', None)
                    token = self._get_access_token()
                    if token and attempt < self.max_retries - 1:
                        continue  # Повторяем попытку
                
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
        Получение эмбеддинга для одного текста через GigaChat API.
        
        Args:
            text: Текст для обработки
            attempt: Номер попытки (для логирования)
        
        Returns:
            Эмбеддинг или None при ошибке
        """
        endpoint = f"{self.api_url}/embeddings"
        
        # Получаем токен доступа
        token = self._get_access_token()
        if not token:
            logger.error("Не удалось получить токен доступа GigaChat")
            return None
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # GigaChat API возвращает эмбеддинг в data[0].embedding
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
                    # Альтернативный формат
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
            # Если токен истек, пытаемся обновить его
            if e.response.status_code == 401:
                logger.warning("Токен доступа истек, обновляем...")
                object.__setattr__(self, 'access_token', None)
                token = self._get_access_token()
                if token and attempt < self.max_retries - 1:
                    # Повторяем запрос с новым токеном
                    return self._get_single_embedding(text, attempt + 1)
            
            logger.error(f"HTTP ошибка {e.response.status_code}: {e.response.text}")
            raise
        
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}", exc_info=True)
            raise

