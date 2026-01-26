"""
Утилита для управления коллекциями Qdrant.

Позволяет очищать коллекции, удалять документы и управлять данными.
"""

import logging
from typing import Optional, List, Dict, Any
from rag.vector_store import QdrantVectorStoreManager
from utils.config import get_config

logger = logging.getLogger(__name__)


class CollectionManager:
    """
    Менеджер для управления коллекциями Qdrant.
    
    Предоставляет методы для очистки коллекций, удаления документов
    и получения статистики.
    """
    
    def __init__(self, vector_store_manager: Optional[QdrantVectorStoreManager] = None):
        """
        Инициализация менеджера коллекций.
        
        Args:
            vector_store_manager: Менеджер векторного хранилища (если None, создается новый)
        """
        if vector_store_manager is None:
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            vector_store_manager = QdrantVectorStoreManager(
                url=qdrant_config.get("url", "http://localhost:6333"),
                api_key=qdrant_config.get("api_key"),
                collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
        
        self.vector_store_manager = vector_store_manager
        logger.info("CollectionManager инициализирован")
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Полная очистка коллекции (удаление всех точек).
        
        Returns:
            Словарь с результатами операции
        """
        try:
            collection_info_before = self.vector_store_manager.get_collection_info()
            points_before = collection_info_before.get("vectors_count", 0)
            
            logger.info(f"Начало очистки коллекции {self.vector_store_manager.collection_name}")
            
            # Получаем все точки для удаления
            # Используем scroll для получения всех точек
            all_points = []
            offset = None
            
            while True:
                scroll_result = self.vector_store_manager.client.scroll(
                    collection_name=self.vector_store_manager.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                
                points_batch = scroll_result[0]
                if not points_batch:
                    break
                
                all_points.extend([point.id for point in points_batch])
                offset = scroll_result[1]  # Следующий offset
                
                if offset is None:
                    break
            
            # Удаляем все точки
            if all_points:
                from qdrant_client.models import PointIdsList
                
                # Удаляем батчами по 1000 точек
                batch_size = 1000
                for i in range(0, len(all_points), batch_size):
                    batch = all_points[i:i + batch_size]
                    self.vector_store_manager.client.delete(
                        collection_name=self.vector_store_manager.collection_name,
                        points_selector=PointIdsList(points=batch)
                    )
                
                logger.info(f"Удалено {len(all_points)} точек из коллекции")
            else:
                logger.info("Коллекция уже пуста")
            
            collection_info_after = self.vector_store_manager.get_collection_info()
            points_after = collection_info_after.get("vectors_count", 0)
            
            result = {
                "success": True,
                "collection_name": self.vector_store_manager.collection_name,
                "points_deleted": points_before - points_after,
                "points_before": points_before,
                "points_after": points_after
            }
            
            logger.info(f"Коллекция очищена: удалено {result['points_deleted']} точек")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "collection_name": self.vector_store_manager.collection_name
            }
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Удаление всех точек конкретного документа.
        
        Args:
            document_id: ID документа для удаления
        
        Returns:
            Словарь с результатами операции
        """
        try:
            logger.info(f"Удаление документа {document_id} из коллекции")
            
            # Поиск всех точек с указанным document_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Получаем список точек для удаления
            scroll_result = self.vector_store_manager.client.scroll(
                collection_name=self.vector_store_manager.collection_name,
                scroll_filter=filter_condition,
                limit=10000,  # Максимальное количество точек для удаления за раз
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = [point.id for point in scroll_result[0]]
            
            if not points_to_delete:
                logger.warning(f"Документ {document_id} не найден в коллекции")
                return {
                    "success": False,
                    "error": "Документ не найден",
                    "document_id": document_id,
                    "points_deleted": 0
                }
            
            # Удаление точек батчами
            from qdrant_client.models import PointIdsList
            
            batch_size = 1000
            deleted_count = 0
            for i in range(0, len(points_to_delete), batch_size):
                batch = points_to_delete[i:i + batch_size]
                self.vector_store_manager.client.delete(
                    collection_name=self.vector_store_manager.collection_name,
                    points_selector=PointIdsList(points=batch)
                )
                deleted_count += len(batch)
            
            result = {
                "success": True,
                "document_id": document_id,
                "points_deleted": deleted_count
            }
            
            logger.info(f"Документ {document_id} удален: {len(points_to_delete)} точек")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при удалении документа {document_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех документов в коллекции.
        
        Returns:
            Список словарей с информацией о документах
        """
        try:
            # Получаем все точки с payload
            scroll_result = self.vector_store_manager.client.scroll(
                collection_name=self.vector_store_manager.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            documents = {}
            
            for point in scroll_result[0]:
                if point.payload:
                    doc_id = point.payload.get("document_id")
                    if doc_id:
                        if doc_id not in documents:
                            documents[doc_id] = {
                                "document_id": doc_id,
                                "document_path": point.payload.get("document_path"),
                                "chunks_count": 0
                            }
                        documents[doc_id]["chunks_count"] += 1
            
            result = list(documents.values())
            
            logger.info(f"Найдено документов: {len(result)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при получении списка документов: {e}", exc_info=True)
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            Словарь со статистикой
        """
        try:
            collection_info = self.vector_store_manager.get_collection_info()
            documents = self.list_documents()
            
            total_chunks = sum(doc["chunks_count"] for doc in documents)
            
            return {
                "collection_name": collection_info["name"],
                "total_points": collection_info["vectors_count"],
                "total_documents": len(documents),
                "total_chunks": total_chunks,
                "status": collection_info.get("status"),
                "config": collection_info.get("config"),
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
            return {
                "error": str(e)
            }
    
    def recreate_collection(self) -> Dict[str, Any]:
        """
        Пересоздание коллекции (удаление и создание заново).
        
        Returns:
            Словарь с результатами операции
        """
        try:
            collection_name = self.vector_store_manager.collection_name
            
            logger.info(f"Пересоздание коллекции {collection_name}")
            
            # Удаление коллекции
            self.vector_store_manager.delete_collection()
            
            # Создание новой коллекции
            self.vector_store_manager.ensure_collection_exists()
            
            result = {
                "success": True,
                "collection_name": collection_name,
                "message": "Коллекция успешно пересоздана"
            }
            
            logger.info(f"Коллекция {collection_name} успешно пересоздана")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при пересоздании коллекции: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "collection_name": self.vector_store_manager.collection_name
            }

