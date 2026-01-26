"""
Основной RAG пайплайн, объединяющий все компоненты системы.
"""

import logging
from typing import List, Dict, Any, Optional

from utils.config import get_config
from rag.chunker_integration import ChunkerIntegration
from rag.giga_embeddings import GigaEmbedding
from rag.vector_store import QdrantVectorStoreManager
from rag.indexer import DocumentIndexer
from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Основной класс RAG-пайплайна.
    
    Объединяет все компоненты системы:
    - Обработку документов через SmartChanker
    - Генерацию эмбеддингов через GigaChat (GigaEmbeddings)
    - Индексацию в Qdrant
    - Поиск и извлечение контекста
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация RAG-пайплайна.
        
        Args:
            config: Словарь конфигурации (если None, загружается из config.yaml)
        """
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Инициализация компонентов
        self._init_components()
        
        logger.info("RAGPipeline инициализирован")
    
    def _init_components(self) -> None:
        """Инициализация всех компонентов пайплайна."""
        # Конфигурация чанкера
        chunker_config = self.config.get("chunker", {})
        self.chunker = ChunkerIntegration(
            chunker_config_path=chunker_config.get("config_path", "config.json"),
            output_dir=chunker_config.get("output_dir", "data/chunks")
        )
        
        # Конфигурация эмбеддингов - используется только GigaEmbeddings
        import os
        embeddings_config = self.config.get("embeddings", {})
        giga_config = embeddings_config.get("giga", {})
        
        # Ключ берется ТОЛЬКО из переменной окружения .env
        credentials = os.getenv("GIGACHAT_AUTH_KEY")
        if not credentials:
            logger.warning(
                "GIGACHAT_AUTH_KEY не найден в переменных окружения. "
                "Убедитесь, что ключ указан в файле .env"
            )
        
        self.embedding = GigaEmbedding(
            credentials=credentials,
            scope=giga_config.get("scope", "GIGACHAT_API_PERS"),
            api_url=giga_config.get("api_url", "https://gigachat.devices.sberbank.ru/api/v1"),
            model=giga_config.get("model", "Embeddings"),
            batch_size=giga_config.get("batch_size", 10),
            max_retries=giga_config.get("max_retries", 3),
            timeout=giga_config.get("timeout", 60)
        )
        
        # Конфигурация векторного хранилища
        qdrant_config = self.config.get("qdrant", {})
        self.vector_store_manager = QdrantVectorStoreManager(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
            vector_size=qdrant_config.get("vector_size", 1024),
            timeout=qdrant_config.get("timeout", 30)
        )
        
        # Создание коллекции, если не существует
        self.vector_store_manager.ensure_collection_exists()
        
        # Инициализация индексера и ретривера
        rag_config = self.config.get("rag", {})
        self.indexer = DocumentIndexer(
            chunker=self.chunker,
            embedding=self.embedding,
            vector_store_manager=self.vector_store_manager
        )
        
        # Конфигурация гибридного поиска
        hybrid_config = rag_config.get("hybrid_search", {})
        
        # Конфигурация реранкера
        reranker_config = rag_config.get("reranker", {})
        reranker = None
        if reranker_config.get("enabled", False):
            try:
                from rag.reranker import ChatCompletionsReranker
                reranker = ChatCompletionsReranker(
                    model=reranker_config.get("model", "dengcao/Qwen3-Reranker-0.6B:F16"),
                    api_url=reranker_config.get("api_url", "http://localhost:11434"),
                    max_retries=reranker_config.get("max_retries", 3),
                    timeout=reranker_config.get("timeout", 60)
                )
                logger.info("Реранкер инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать реранкер: {e}")
                reranker = None
        
        self.retriever = DocumentRetriever(
            embedding=self.embedding,
            vector_store_manager=self.vector_store_manager,
            top_k=rag_config.get("top_k", 5),
            hybrid_search_enabled=hybrid_config.get("enabled", True),
            vector_top_k=hybrid_config.get("vector_top_k", 20),
            text_top_k=hybrid_config.get("text_top_k", 20),
            reranker=reranker
        )
    
    def index_document(
        self,
        document_path: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Индексация документа.
        
        Args:
            document_path: Путь к документу
            document_id: ID документа
            metadata: Дополнительные метаданные
        
        Returns:
            Результат индексации
        """
        return self.indexer.index_document(document_path, document_id, metadata)
    
    def index_folder(
        self,
        folder_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Индексация всех документов в папке.
        
        Args:
            folder_path: Путь к папке
            metadata: Дополнительные метаданные
        
        Returns:
            Список результатов индексации
        """
        return self.indexer.index_folder(folder_path, metadata)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            filter_metadata: Фильтр по метаданным
        
        Returns:
            Список найденных чанков
        """
        return self.retriever.retrieve(query, top_k, filter_metadata)
    
    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Получение контекста для запроса.
        
        Args:
            query: Текст запроса
            top_k: Количество чанков
        
        Returns:
            Текст контекста
        """
        return self.retriever.get_context(query, top_k)
    
    def get_context_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Получение контекста с метаданными.
        
        Args:
            query: Текст запроса
            top_k: Количество чанков
        
        Returns:
            Словарь с контекстом и метаданными
        """
        return self.retriever.get_context_with_metadata(query, top_k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получение информации о коллекции.
        
        Returns:
            Информация о коллекции
        """
        return self.vector_store_manager.get_collection_info()
    
    def get_indexed_documents_count(self) -> int:
        """
        Получение количества проиндексированных документов.
        
        Returns:
            Количество точек в коллекции
        """
        return self.indexer.get_indexed_documents_count()

