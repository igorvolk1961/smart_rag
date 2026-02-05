"""
Кастомный эмбеддинг-класс для GigaChat API (GigaEmbeddings).
"""

import logging
import time
import asyncio
import uuid
from typing import List, Optional
from datetime import datetime
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
            credentials: Base64-encoded Authorization Key в формате base64(client_id:client_secret)
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
        object.__setattr__(self, 'token_obtained_at', None)  # Время получения токена
        
        logger.info(
            f"GigaEmbedding инициализирован: model={model}, "
            f"api_url={api_url}, batch_size={batch_size}, embedding_dim={self.embedding_dim}"
        )
        
        # Получаем токен доступа при инициализации
        self._get_access_token()
    
    def _get_access_token(self) -> Optional[str]:
        """
        Получение токена доступа для GigaChat API через Authorization Key (OAuth2).
        
        Используется только метод получения токена через base64-encoded Authorization Key.
        Формат ключа: base64(client_id:client_secret)
        
        Токен кэшируется и действителен в течение 30 минут (согласно документации GigaChat).
        Лимит запросов на получение токена: до 10 раз в секунду.
        
        Returns:
            Токен доступа
            
        Raises:
            RuntimeError: Если не удалось получить токен
            ValueError: Если формат Authorization Key неверный
        """
        # Проверяем, есть ли токен и не истек ли он (действителен 30 минут)
        if self.access_token and self.token_obtained_at:
            # Проверяем, не истек ли токен (30 минут = 1800 секунд)
            token_age = (datetime.now() - self.token_obtained_at).total_seconds()
            if token_age < 1800:  # Токен еще действителен
                logger.debug(f"Используется кэшированный токен (возраст: {token_age:.0f} сек)")
                return self.access_token
            else:
                logger.info(f"Токен истек (возраст: {token_age:.0f} сек, лимит: 1800 сек), запрашиваем новый")
                object.__setattr__(self, 'access_token', None)
                object.__setattr__(self, 'token_obtained_at', None)
        
        # Проверяем наличие credentials (Authorization Key)
        if not self.credentials:
            error_msg = (
                "Не указан Authorization Key для GigaChat API. "
                "Укажите base64-encoded строку в формате base64(client_id:client_secret) "
                "в параметре credentials или переменной окружения GIGACHAT_AUTH_KEY."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Получаем токен напрямую через OAuth2 из Authorization Key
        token = self._get_token_from_key(self.credentials)
        if not token:
            error_msg = "Не удалось получить токен доступа из Authorization Key"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Сохраняем токен и время его получения
        object.__setattr__(self, 'access_token', token)
        object.__setattr__(self, 'token_obtained_at', datetime.now())
        logger.info("Токен доступа GigaChat получен из Authorization Key (действителен 30 минут)")
        return token
    
    def _generate_rquid(self) -> str:
        """
        Генерация уникального идентификатора запроса (RqUID) для GigaChat API.
        
        Returns:
            UUID строки в формате, требуемом GigaChat API
        """
        return str(uuid.uuid4())
    
    def _get_token_from_key(self, auth_key: str) -> Optional[str]:
        """
        Получение токена доступа напрямую из Authorization Key через OAuth2.
        
        Согласно документации GigaChat API и стандарту OAuth2 client credentials flow:
        - Authorization Key (base64-encoded client_id:client_secret) декодируется
        - client_id и client_secret используются для Basic authentication
        - Для OAuth2 client credentials flow требуется grant_type=client_credentials
        - RqUID (Request UID) - уникальный идентификатор запроса
        
        Args:
            auth_key: Base64-encoded Authorization Key (client_id:client_secret)
        
        Returns:
            Токен доступа
            
        Raises:
            ValueError: Если формат Authorization Key неверный
            RuntimeError: Если не удалось получить токен
        """
        try:
            import base64
            
            # URL для получения токена
            token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

#            auth_key1 = "M2RjNGFkZGEtOTA0MS00MzI0LTlmNzUtNzczNTIxNmQ0Zjk1OmNiMDcxYTkyLTE5MTctNDk2MS1hOWZjLTIwMjgxZDU1NWUxZg=="

            headers = {
                "Authorization": f"Basic {auth_key}",  # Authorization Key передается напрямую
                "RqUID": self._generate_rquid(),  # Уникальный идентификатор запроса
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            
            # Для OAuth2 client credentials flow требуется grant_type
            payload = {"scope":self.scope}

            # Отправляем запрос на получение токена
            with httpx.Client(timeout=self.timeout, verify=False) as client:
                response = client.post(token_url, headers=headers, data=payload)
                
                # Логируем детали ошибки для отладки
                if response.status_code != 200:
                    logger.error(
                        f"Ошибка получения токена GigaChat: статус {response.status_code}, "
                        f"заголовки: {dict(response.headers)}, тело: {response.text}"
                    )
                
                response.raise_for_status()
                
                token_data = response.json()
                
                if "access_token" in token_data:
                    return token_data["access_token"]
                else:
                    error_msg = f"Токен не найден в ответе OAuth2 сервера GigaChat. Доступные поля: {token_data.keys()}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except ValueError:
            # Пробрасываем ValueError (ошибка декодирования ключа) дальше
            raise
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP ошибка при получении токена доступа GigaChat: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Ошибка при получении токена из Authorization Key: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
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
                        raise result  # Пробрасываем исключение дальше
                    if result:
                        embeddings.append(result)
                    else:
                        # Это не должно произойти, так как _aget_single_embedding теперь выбрасывает исключения
                        error_msg = f"Не удалось получить эмбеддинг для текста {i}: метод вернул None"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
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
            error_msg = "Не удалось получить токен доступа GigaChat. Проверьте настройки авторизации (GIGACHAT_AUTH_KEY или credentials)."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
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
                            error_msg = f"Неверный формат эмбеддинга в ответе GigaChat API: ожидался список, получен {type(embedding)}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"Поле 'embedding' отсутствует в data[0] ответа GigaChat API. Доступные поля: {embedding_data.keys()}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                elif "embedding" in data:
                    # Альтернативный формат
                    embedding = data["embedding"]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    else:
                        error_msg = f"Неверный формат эмбеддинга в ответе GigaChat API (альтернативный формат): ожидался список, получен {type(embedding)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                else:
                    error_msg = f"Поле 'embedding' или 'data' отсутствует в ответе GigaChat API. Доступные поля: {data.keys()}"
                    logger.error(error_msg)
                    logger.debug(f"Полный ответ: {data}")
                    raise ValueError(error_msg)
                    
        except httpx.TimeoutException:
            logger.warning(f"Таймаут при получении эмбеддинга для текста (длина: {len(text)})")
            raise
        
        except httpx.HTTPStatusError as e:
            # Если токен истек, пытаемся обновить его
            if e.response.status_code == 401:
                logger.warning("Токен доступа истек (401), обновляем...")
                object.__setattr__(self, 'access_token', None)
                object.__setattr__(self, 'token_obtained_at', None)
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
                    try:
                        embedding = self._get_single_embedding(text, attempt)
                        if embedding:
                            embeddings.append(embedding)
                        else:
                            # Это не должно произойти, так как _get_single_embedding теперь выбрасывает исключения
                            error_msg = "Не удалось получить эмбеддинг для текста: метод вернул None"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                    except (RuntimeError, ValueError) as e:
                        # Ошибка получения токена доступа или неверный формат ответа - пробрасываем дальше
                        logger.error(f"Ошибка при получении эмбеддинга: {e}")
                        raise
                
                return embeddings
                
            except (RuntimeError, ValueError):
                # Ошибка получения токена доступа или неверный формат ответа - пробрасываем дальше без retry
                raise
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
                    logger.warning("Токен доступа истек (401), обновляем...")
                    object.__setattr__(self, 'access_token', None)
                    object.__setattr__(self, 'token_obtained_at', None)
                    try:
                        token = self._get_access_token()
                        if not token:
                            # Это не должно произойти, так как _get_access_token теперь выбрасывает исключения
                            error_msg = "Не удалось получить новый токен доступа после истечения старого"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        if attempt < self.max_retries - 1:
                            continue  # Повторяем попытку с новым токеном
                    except (RuntimeError, ValueError):
                        # Ошибка получения нового токена - пробрасываем дальше
                        raise
                
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
            error_msg = "Не удалось получить токен доступа GigaChat. Проверьте настройки авторизации (GIGACHAT_AUTH_KEY или credentials)."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        payload = {
            "model": self.model,
            "input": text
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        try:
            with httpx.Client(timeout=self.timeout, verify=False) as client:
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
                            error_msg = f"Неверный формат эмбеддинга в ответе GigaChat API: ожидался список, получен {type(embedding)}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    else:
                        error_msg = f"Поле 'embedding' отсутствует в data[0] ответа GigaChat API. Доступные поля: {embedding_data.keys()}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                elif "embedding" in data:
                    # Альтернативный формат
                    embedding = data["embedding"]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    else:
                        error_msg = f"Неверный формат эмбеддинга в ответе GigaChat API (альтернативный формат): ожидался список, получен {type(embedding)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                else:
                    error_msg = f"Поле 'embedding' или 'data' отсутствует в ответе GigaChat API. Доступные поля: {data.keys()}"
                    logger.error(error_msg)
                    logger.debug(f"Полный ответ: {data}")
                    raise ValueError(error_msg)
                    
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

