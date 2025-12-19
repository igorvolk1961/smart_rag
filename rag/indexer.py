"""
Модуль для индексации документов в векторное хранилище.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core.schema import TextNode

from rag.chunker_integration import ChunkerIntegration
from rag.embeddings import OllamaEmbedding
from rag.vector_store import QdrantVectorStoreManager

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Класс для индексации документов в векторное хранилище.
    
    Использует SmartChanker для обработки документов и создания чанков,
    затем генерирует эмбеддинги и сохраняет их в Qdrant.
    """
    
    def __init__(
        self,
        chunker: ChunkerIntegration,
        embedding: OllamaEmbedding,
        vector_store_manager: QdrantVectorStoreManager
    ):
        """
        Инициализация индексера документов.
        
        Args:
            chunker: Интеграция SmartChanker
            embedding: Эмбеддинг-модель
            vector_store_manager: Менеджер векторного хранилища
        """
        self.chunker = chunker
        self.embedding = embedding
        self.vector_store_manager = vector_store_manager
        
        # Получаем векторное хранилище для LlamaIndex
        self.vector_store = vector_store_manager.get_vector_store()
        
        logger.info("DocumentIndexer инициализирован")
    
    def index_document(
        self,
        document_path: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Индексация одного документа.
        
        Args:
            document_path: Путь к документу
            document_id: Уникальный ID документа (если None, генерируется)
            metadata: Дополнительные метаданные документа
        
        Returns:
            Словарь с результатами индексации
        """
        if document_id is None:
            document_id = Path(document_path).stem
        
        logger.info(f"Начало индексации документа: {document_path} (ID: {document_id})")
        
        # Обработка документа через SmartChanker
        chunker_result = self.chunker.process_document(document_path, document_id)
        
        # Подготовка метаданных
        doc_metadata = {
            "document_id": document_id,
            "document_path": str(document_path),
            **(metadata or {})
        }
        
        # Создание узлов из чанков
        nodes = self._create_nodes_from_chunks(
            chunker_result["chunks"],
            doc_metadata
        )
        
        # Генерация эмбеддингов и сохранение в векторное хранилище
        indexed_count = self._index_nodes(nodes)
        
        result = {
            "document_id": document_id,
            "document_path": str(document_path),
            "chunks_processed": len(chunker_result["chunks"]),
            "nodes_indexed": indexed_count,
            "metadata": doc_metadata
        }
        
        logger.info(
            f"Документ {document_id} проиндексирован: "
            f"{indexed_count} узлов из {len(chunker_result['chunks'])} чанков"
        )
        
        return result
    
    def _create_nodes_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any]
    ) -> List[TextNode]:
        """
        Создание узлов LlamaIndex из чанков SmartChanker.
        
        Args:
            chunks: Список чанков из SmartChanker
            document_metadata: Метаданные документа
        
        Returns:
            Список узлов TextNode
        """
        nodes = []
        
        for idx, chunk_data in enumerate(chunks):
            text = chunk_data.get("text", "")
            if not text or not text.strip():
                continue
            
            # Объединяем метаданные документа и чанка
            chunk_metadata = chunk_data.get("metadata", {})
            node_metadata = {
                **document_metadata,
                **chunk_metadata,
                "chunk_index": idx,
                "chunk_id": f"{document_metadata['document_id']}_chunk_{idx}"
            }
            
            # Создание узла
            node = TextNode(
                text=text.strip(),
                metadata=node_metadata,
                id_=node_metadata["chunk_id"]
            )
            
            nodes.append(node)
        
        logger.debug(f"Создано {len(nodes)} узлов из {len(chunks)} чанков")
        
        return nodes
    
    def _index_nodes(self, nodes: List[TextNode]) -> int:
        """
        Индексация узлов в векторное хранилище.
        
        Args:
            nodes: Список узлов для индексации
        
        Returns:
            Количество проиндексированных узлов
        """
        if not nodes:
            logger.warning("Нет узлов для индексации")
            return 0
        
        try:
            # Генерация эмбеддингов для всех узлов
            texts = [node.text for node in nodes]
            # Используем приватный метод напрямую, так как публичный может быть недоступен из-за Pydantic
            embeddings = self.embedding._get_text_embeddings(texts)
            
            if len(embeddings) != len(nodes):
                logger.error(
                    f"Несоответствие количества эмбеддингов ({len(embeddings)}) "
                    f"и узлов ({len(nodes)})"
                )
                return 0
            
            # Сохранение узлов в векторное хранилище
            from qdrant_client.models import PointStruct
            
            points = []
            for node, embedding in zip(nodes, embeddings):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": node.text,
                        **node.metadata
                    }
                )
                points.append(point)
            
            # Вставка точек в Qdrant
            self.vector_store_manager.client.upsert(
                collection_name=self.vector_store_manager.collection_name,
                points=points
            )
            
            logger.info(f"Успешно проиндексировано {len(points)} узлов")
            
            return len(points)
            
        except Exception as e:
            logger.error(f"Ошибка при индексации узлов: {e}", exc_info=True)
            raise
    
    def index_folder(
        self,
        folder_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Индексация всех документов в папке.
        
        Args:
            folder_path: Путь к папке с документами
            metadata: Дополнительные метаданные для всех документов
        
        Returns:
            Список результатов индексации для каждого документа
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        
        results = []
        supported_formats = [".docx", ".txt", ".pdf"]
        
        for doc_file in folder.iterdir():
            if doc_file.is_file() and doc_file.suffix.lower() in supported_formats:
                try:
                    result = self.index_document(str(doc_file), metadata=metadata)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при индексации {doc_file}: {e}", exc_info=True)
        
        logger.info(f"Проиндексировано документов: {len(results)} из {len(list(folder.iterdir()))}")
        
        return results
    
    def get_indexed_documents_count(self) -> int:
        """
        Получение количества проиндексированных документов.
        
        Returns:
            Количество точек в коллекции
        """
        return self.vector_store_manager.get_points_count()

