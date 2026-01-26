"""
Модуль для поиска и извлечения контекста из векторного хранилища.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue

from rag.giga_embeddings import GigaEmbedding
from rag.vector_store import QdrantVectorStoreManager

logger = logging.getLogger(__name__)

# Импорт реранкера (опциональный, чтобы избежать циклических зависимостей)
try:
    from rag.reranker import ChatCompletionsReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    ChatCompletionsReranker = None


class DocumentRetriever:
    """
    Класс для поиска и извлечения релевантных чанков из векторного хранилища.
    
    Поддерживает гибридный поиск: комбинацию векторного и полнотекстового поиска.
    """
    
    def __init__(
        self,
        embedding: GigaEmbedding,
        vector_store_manager: QdrantVectorStoreManager,
        top_k: int = 5,
        hybrid_search_enabled: bool = True,
        vector_top_k: int = 20,
        text_top_k: int = 20,
        reranker: Optional[Any] = None
    ):
        """
        Инициализация ретривера документов.
        
        Args:
            embedding: Эмбеддинг-модель для запросов
            vector_store_manager: Менеджер векторного хранилища
            top_k: Финальное количество топ-результатов для извлечения (после реранкера)
            hybrid_search_enabled: Включить гибридный поиск
            vector_top_k: Количество документов для векторного поиска
            text_top_k: Количество документов для полнотекстового поиска
            reranker: Опциональный реранкер для переупорядочивания результатов
        """
        self.embedding = embedding
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k
        self.hybrid_search_enabled = hybrid_search_enabled
        self.vector_top_k = vector_top_k
        self.text_top_k = text_top_k
        self.reranker = reranker
        
        logger.info(
            f"DocumentRetriever инициализирован: top_k={top_k}, "
            f"hybrid_search={hybrid_search_enabled}, "
            f"vector_top_k={vector_top_k}, text_top_k={text_top_k}, "
            f"reranker={'enabled' if reranker else 'disabled'}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков по запросу (гибридный поиск).
        
        Args:
            query: Текст запроса
            top_k: Количество результатов (если None, используется значение по умолчанию)
            filter_metadata: Фильтр по метаданным (например, {"document_id": "doc1"})
        
        Returns:
            Список словарей с найденными чанками и метаданными
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Поиск по запросу: '{query[:50]}...' (top_k={top_k}, hybrid={self.hybrid_search_enabled})")
        
        # Выполняем поиск (гибридный или векторный)
        if self.hybrid_search_enabled:
            results = self._hybrid_search(query, top_k, filter_metadata)
        else:
            results = self._vector_search(query, top_k, filter_metadata)
        
        # Применяем реранкинг, если он включен
        if self.reranker and results:
            logger.debug(f"Применение реранкера к {len(results)} результатам")
            # Реранкер обрабатывает все документы и возвращает отсортированные результаты
            results = self.reranker.rerank(query, results, top_k=top_k)
            logger.debug(f"После реранкинга: {len(results)} результатов")
        
        return results[:top_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Векторный поиск по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков
        """
        try:
            # Генерация эмбеддинга для запроса
            query_embedding = self.embedding._get_query_embedding(query)
            
            if not query_embedding:
                logger.error("Не удалось получить эмбеддинг для запроса")
                return []
            
            # Подготовка фильтра по метаданным
            search_filter = self._prepare_filter(filter_metadata)
            
            # Поиск в Qdrant через HTTP API
            import httpx
            
            search_url = f"{self.vector_store_manager.url}/collections/{self.vector_store_manager.collection_name}/points/search"
            
            filter_dict = self._filter_to_dict(search_filter)
            
            response = httpx.post(
                search_url,
                json={
                    "vector": query_embedding,
                    "filter": filter_dict,
                    "limit": top_k,
                    "with_payload": True,
                    "with_vector": False
                },
                timeout=30
            )
            response.raise_for_status()
            search_results = response.json().get("result", [])
            
            return self._parse_search_results(search_results)
            
        except Exception as e:
            logger.error(f"Ошибка при векторном поиске: {e}", exc_info=True)
            return []
    
    def _text_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Полнотекстовый поиск по запросу через Qdrant.
        
        Использует Query API с текстовым запросом или fallback на scroll с фильтрацией.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков
        """
        # Пробуем использовать Query API (если поддерживается)
        try:
            return self._text_search_query_api(query, top_k, filter_metadata)
        except Exception as e:
            logger.debug(f"Query API не поддерживается, используем fallback: {e}")
            # Fallback на простой поиск через scroll с фильтрацией по тексту
            return self._text_search_fallback(query, top_k, filter_metadata)
    
    def _text_search_query_api(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Полнотекстовый поиск через Qdrant Query API.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков
        """
        import httpx
        
        # Подготовка фильтра по метаданным
        search_filter = self._prepare_filter(filter_metadata)
        filter_dict = self._filter_to_dict(search_filter)
        
        # Qdrant Query API для полнотекстового поиска
        query_url = f"{self.vector_store_manager.url}/collections/{self.vector_store_manager.collection_name}/points/query"
        
        # Формируем запрос для полнотекстового поиска
        # В Qdrant полнотекстовый поиск может работать через query с текстом
        query_payload = {
            "query": {
                "text": query  # Полнотекстовый поиск по полю "text" в payload
            },
            "filter": filter_dict,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False
        }
        
        response = httpx.post(query_url, json=query_payload, timeout=30)
        response.raise_for_status()
        
        query_results = response.json().get("result", {})
        
        # Qdrant Query API возвращает результаты в формате {"points": [...]}
        if isinstance(query_results, dict) and "points" in query_results:
            search_results = query_results["points"]
        elif isinstance(query_results, list):
            search_results = query_results
        else:
            raise ValueError(f"Неожиданный формат результатов: {type(query_results)}")
        
        # Преобразуем результаты в формат с score
        results = []
        for idx, result in enumerate(search_results):
            if isinstance(result, dict):
                payload = result.get('payload', {})
                result_id = result.get('id')
                # Используем обратный ранг как score
                score = 1.0 / (idx + 1)
            else:
                payload = result.payload if result.payload else {}
                result_id = result.id
                score = 1.0 / (idx + 1)
            
            chunk_data = {
                "text": payload.get("text", ""),
                "score": score,
                "id": result_id,
                "metadata": {
                    k: v for k, v in payload.items() if k != "text"
                }
            }
            results.append(chunk_data)
        
        logger.debug(f"Query API полнотекстовый поиск вернул {len(results)} результатов")
        return results
    
    def _text_search_fallback(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback метод полнотекстового поиска через scroll с фильтрацией.
        
        Использует простой поиск по подстроке в тексте через scroll и фильтрацию.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков
        """
        try:
            # Подготовка фильтра
            search_filter = self._prepare_filter(filter_metadata)
            
            # Получаем все точки через scroll (с ограничением)
            scroll_results = self.vector_store_manager.client.scroll(
                collection_name=self.vector_store_manager.collection_name,
                scroll_filter=search_filter,
                limit=min(top_k * 10, 1000),  # Берем больше для фильтрации
                with_payload=True,
                with_vectors=False
            )
            
            # Фильтруем по тексту (простой поиск подстроки)
            query_lower = query.lower()
            matched_results = []
            
            for point in scroll_results[0]:
                payload = point.payload if point.payload else {}
                text = payload.get("text", "")
                
                # Простой поиск подстроки в тексте
                if query_lower in text.lower():
                    # Вычисляем простую оценку релевантности (количество вхождений)
                    text_lower = text.lower()
                    occurrences = text_lower.count(query_lower)
                    # Нормализуем по длине текста
                    score = occurrences / max(len(text_lower), 1)
                    
                    chunk_data = {
                        "text": text,
                        "score": score,
                        "id": point.id,
                        "metadata": {
                            k: v for k, v in payload.items() if k != "text"
                        }
                    }
                    matched_results.append(chunk_data)
            
            # Сортируем по score и берем топ-k
            matched_results.sort(key=lambda x: x["score"], reverse=True)
            results = matched_results[:top_k]
            
            logger.debug(f"Fallback полнотекстовый поиск вернул {len(results)} результатов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при fallback полнотекстовом поиске: {e}", exc_info=True)
            return []
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: объединение векторного и полнотекстового поиска.
        
        Просто объединяет результаты двух поисков, убирая дубликаты.
        Ранжирование выполняется реранкером.
        
        Args:
            query: Текст запроса
            top_k: Финальное количество результатов (не используется здесь)
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков без дубликатов
        """
        # Выполняем оба типа поиска с явно заданными параметрами
        vector_results = self._vector_search(query, self.vector_top_k, filter_metadata)
        text_results = self._text_search(query, self.text_top_k, filter_metadata)
        
        logger.debug(
            f"Гибридный поиск: векторный={len(vector_results)} (top_k={self.vector_top_k}), "
            f"полнотекстовый={len(text_results)} (top_k={self.text_top_k})"
        )
        
        # Объединяем результаты, убирая дубликаты по ID
        # Это один и тот же документ, различается только score, который реранкер переоценит
        combined_results = {}
        
        # Добавляем все результаты, убирая дубликаты
        for result in vector_results + text_results:
            result_id = result.get("id")
            if result_id and result_id not in combined_results:
                combined_results[result_id] = result
        
        # Преобразуем в список
        merged_results = list(combined_results.values())
        
        logger.debug(
            f"Объединено результатов: {len(merged_results)} (было: векторный={len(vector_results)}, "
            f"полнотекстовый={len(text_results)})"
        )
        
        return merged_results
    
    def _prepare_filter(self, filter_metadata: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
        """Подготовка фильтра по метаданным."""
        if not filter_metadata:
            return None
        
        must_conditions = []
        for key, value in filter_metadata.items():
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        
        if must_conditions:
            return Filter(must=must_conditions)
        return None
    
    def _filter_to_dict(self, search_filter: Optional[Filter]) -> Optional[Dict]:
        """Преобразование Filter в словарь для JSON."""
        if not search_filter:
            return None
        
        if hasattr(search_filter, 'dict'):
            return search_filter.dict()
        elif hasattr(search_filter, 'model_dump'):
            return search_filter.model_dump()
        elif hasattr(search_filter, 'must') and search_filter.must:
            return {
                "must": [
                    {
                        "key": cond.key,
                        "match": {"value": cond.match.value} if hasattr(cond.match, 'value') else {"value": cond.match}
                    }
                    for cond in search_filter.must
                    if hasattr(cond, 'key') and hasattr(cond, 'match')
                ]
            }
        return None
    
    def _parse_search_results(self, search_results: List) -> List[Dict[str, Any]]:
        """Парсинг результатов поиска в единый формат."""
        results = []
        
        if isinstance(search_results, list):
            for result in search_results:
                if hasattr(result, 'payload'):
                    payload = result.payload if result.payload else {}
                    score = getattr(result, 'score', 0.0)
                    result_id = result.id
                elif isinstance(result, dict):
                    payload = result.get('payload', {})
                    score = result.get('score', 0.0)
                    result_id = result.get('id')
                else:
                    logger.warning(f"Неожиданный формат результата: {type(result)}")
                    continue
                
                chunk_data = {
                    "text": payload.get("text", ""),
                    "score": score,
                    "id": result_id,
                    "metadata": {
                        k: v for k, v in payload.items() if k != "text"
                    }
                }
                results.append(chunk_data)
        elif hasattr(search_results, 'points'):
            for point in search_results.points:
                payload = point.payload if point.payload else {}
                score = getattr(point, 'score', 0.0)
                result_id = point.id
                
                chunk_data = {
                    "text": payload.get("text", ""),
                    "score": score,
                    "id": result_id,
                    "metadata": {
                        k: v for k, v in payload.items() if k != "text"
                    }
                }
                results.append(chunk_data)
        else:
            logger.warning(f"Неожиданный формат результатов поиска: {type(search_results)}")
        
        return results
    
    def retrieve_by_document_id(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск чанков в конкретном документе.
        
        Args:
            document_id: ID документа
            query: Текст запроса
            top_k: Количество результатов
        
        Returns:
            Список найденных чанков
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={"document_id": document_id}
        )
    
    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        Получение контекста из найденных чанков.
        
        Args:
            query: Текст запроса
            top_k: Количество чанков для включения в контекст
            separator: Разделитель между чанками
        
        Returns:
            Объединенный текст контекста
        """
        chunks = self.retrieve(query, top_k=top_k)
        
        if not chunks:
            return ""
        
        # Объединение текстов чанков
        context_parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                context_parts.append(text)
        
        context = separator.join(context_parts)
        
        logger.debug(f"Сформирован контекст длиной {len(context)} символов из {len(chunks)} чанков")
        
        return context
    
    def get_context_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> Dict[str, Any]:
        """
        Получение контекста с метаданными.
        
        Args:
            query: Текст запроса
            top_k: Количество чанков
            separator: Разделитель между чанками
        
        Returns:
            Словарь с контекстом и метаданными
        """
        chunks = self.retrieve(query, top_k=top_k)
        
        if not chunks:
            return {
                "context": "",
                "chunks": [],
                "metadata": {}
            }
        
        # Объединение текстов
        context_parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                context_parts.append(text)
        
        context = separator.join(context_parts)
        
        # Извлечение метаданных
        documents = set()
        for chunk in chunks:
            doc_id = chunk.get("metadata", {}).get("document_id")
            if doc_id:
                documents.add(doc_id)
        
        return {
            "context": context,
            "chunks": chunks,
            "metadata": {
                "chunks_count": len(chunks),
                "documents": list(documents),
                "context_length": len(context)
            }
        }

