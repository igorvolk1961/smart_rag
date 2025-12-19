"""
Модуль для поиска и извлечения контекста из векторного хранилища.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue

from rag.embeddings import OllamaEmbedding
from rag.vector_store import QdrantVectorStoreManager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Класс для поиска и извлечения релевантных чанков из векторного хранилища.
    
    Выполняет векторный поиск по запросу и возвращает наиболее релевантные чанки.
    """
    
    def __init__(
        self,
        embedding: OllamaEmbedding,
        vector_store_manager: QdrantVectorStoreManager,
        top_k: int = 5
    ):
        """
        Инициализация ретривера документов.
        
        Args:
            embedding: Эмбеддинг-модель для запросов
            vector_store_manager: Менеджер векторного хранилища
            top_k: Количество топ-результатов для извлечения
        """
        self.embedding = embedding
        self.vector_store_manager = vector_store_manager
        self.top_k = top_k
        
        logger.info(f"DocumentRetriever инициализирован: top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов (если None, используется значение по умолчанию)
            filter_metadata: Фильтр по метаданным (например, {"document_id": "doc1"})
        
        Returns:
            Список словарей с найденными чанками и метаданными
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Поиск по запросу: '{query[:50]}...' (top_k={top_k})")
        
        try:
            # Генерация эмбеддинга для запроса
            # Используем приватный метод напрямую, так как публичный может быть недоступен из-за Pydantic
            query_embedding = self.embedding._get_query_embedding(query)
            
            if not query_embedding:
                logger.error("Не удалось получить эмбеддинг для запроса")
                return []
            
            # Подготовка фильтра по метаданным
            search_filter = None
            if filter_metadata:
                must_conditions = []
                for key, value in filter_metadata.items():
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                
                if must_conditions:
                    search_filter = Filter(must=must_conditions)
            
            # Поиск в Qdrant
            # Используем прямой вызов через HTTP API (наиболее надежный способ)
            import httpx
            
            try:
                # Формируем URL для поиска
                search_url = f"{self.vector_store_manager.url}/collections/{self.vector_store_manager.collection_name}/points/search"
                
                # Подготавливаем фильтр для запроса
                filter_dict = None
                if search_filter:
                    # Преобразуем Filter в словарь для JSON
                    if hasattr(search_filter, 'dict'):
                        filter_dict = search_filter.dict()
                    elif hasattr(search_filter, 'model_dump'):
                        filter_dict = search_filter.model_dump()
                    else:
                        # Создаем фильтр вручную
                        if hasattr(search_filter, 'must') and search_filter.must:
                            filter_dict = {
                                "must": [
                                    {
                                        "key": cond.key,
                                        "match": {"value": cond.match.value} if hasattr(cond.match, 'value') else {"value": cond.match}
                                    }
                                    for cond in search_filter.must
                                    if hasattr(cond, 'key') and hasattr(cond, 'match')
                                ]
                            }
                
                # Выполняем поиск
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
                
            except Exception as e:
                logger.error(f"Ошибка при поиске в Qdrant: {e}", exc_info=True)
                raise
            
            # Формирование результатов
            results = []
            
            # Обрабатываем результаты поиска
            if isinstance(search_results, list):
                for result in search_results:
                    # Обрабатываем разные форматы результата
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
                # Если результат имеет атрибут points
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
            
            logger.info(f"Найдено {len(results)} релевантных чанков")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}", exc_info=True)
            return []
    
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

