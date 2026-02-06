"""
RAG Tool для поиска в базе знаний.
Реализация RAG поиска с использованием DocumentRetriever.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Dict, Any, ClassVar

from pydantic import Field

from api.agents.base_tool import BaseTool
from api.exceptions import ServiceError

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGTool(BaseTool):
    """Search the knowledge base for relevant information from indexed documents.
    
    This tool searches through the vector database to find relevant chunks of text
    from previously indexed documents. Use this when you need information from
    the user's knowledge base or uploaded documents.
    
    The tool uses hybrid search (vector + full-text) and optional reranking
    to find the most relevant information for your query.
    """

    reasoning: str = Field(description="Why this RAG search is needed and what information is expected")
    query: str = Field(description="Search query to find relevant information in the knowledge base")
    max_results: int = Field(
        description="Maximum number of results to retrieve",
        default=5,
        ge=1,
    )

    # Кэш для переиспользуемых компонентов (на уровне класса)
    _embedding_cache: ClassVar[Dict[str, Any]] = {}
    _vector_store_cache: ClassVar[Dict[str, Any]] = {}
    _retriever_cache: ClassVar[Dict[str, Any]] = {}
    _config_cache: ClassVar[Any] = None

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        """Execute RAG search using DocumentRetriever."""
        
        logger.info(f"🔍 RAG search query: '{self.query}' (max_results={self.max_results})")
        
        try:
            # Получаем параметры из custom_context
            vdb_url = None
            embed_api_key = None
            embed_url = None
            embed_model_name = None
            embed_batch_size = None
            
            # Безопасно получаем custom_context из AgentContext (BaseModel)
            # Используем model_dump() для безопасного извлечения значений полей
            try:
                # Пробуем получить custom_context через model_dump
                # Если возникает ошибка итерации, это означает что custom_context содержит ModelPrivateAttr
                try:
                    context_dict = context.model_dump()
                    custom_context_value = context_dict.get("custom_context")
                except TypeError as dump_error:
                    # Если model_dump() не может обработать custom_context из-за ModelPrivateAttr
                    if "not iterable" in str(dump_error).lower():
                        # Исключаем custom_context из дампа и пропускаем его
                        try:
                            context.model_dump(exclude={"custom_context"})
                            custom_context_value = None
                            logger.debug("custom_context содержит ModelPrivateAttr, пропускаем извлечение параметров")
                        except Exception:
                            custom_context_value = None
                    else:
                        raise  # Перебрасываем другие TypeError
                
                # Обрабатываем полученное значение custom_context
                if custom_context_value is not None:
                    custom_dict = None
                    
                    # Если это словарь - используем напрямую
                    if isinstance(custom_context_value, dict):
                        custom_dict = custom_context_value
                    # Если это BaseModel (проверяем через наличие метода model_dump) - конвертируем в словарь
                    elif hasattr(custom_context_value, "model_dump") and callable(getattr(custom_context_value, "model_dump", None)):
                        try:
                            custom_dict = custom_context_value.model_dump()
                            if not isinstance(custom_dict, dict):
                                custom_dict = None
                        except (TypeError, AttributeError) as e:
                            if "not iterable" in str(e).lower():
                                logger.debug(f"custom_context содержит вложенный ModelPrivateAttr")
                            custom_dict = None
                    
                    # Извлекаем параметры из словаря
                    if isinstance(custom_dict, dict):
                        vdb_url = custom_dict.get("vdb_url")
                        embed_api_key = custom_dict.get("embed_api_key")
                        embed_url = custom_dict.get("embed_url")
                        embed_model_name = custom_dict.get("embed_model_name")
                        embed_batch_size = custom_dict.get("embed_batch_size")
                        
            except TypeError as te:
                # Обрабатываем ошибки итерации (ModelPrivateAttr)
                if "not iterable" in str(te).lower():
                    logger.debug(f"Ошибка итерации при извлечении custom_context (ModelPrivateAttr): {te}")
                else:
                    logger.warning(f"TypeError при извлечении параметров из custom_context: {te}")
            except Exception as e:
                logger.warning(f"Ошибка при извлечении параметров из custom_context: {e}. Используем значения по умолчанию.")
            
            # Проверяем наличие обязательных параметров
            if not vdb_url:
                return self._format_error(
                    "VDB URL не указан",
                    "Для выполнения RAG поиска необходимо указать vdb_url в параметрах запроса или custom_context"
                )
            
            # Инициализируем компоненты RAG
            embedding, vector_store_manager = self._initialize_rag_components(
                vdb_url=vdb_url,
                embed_api_key=embed_api_key,
                embed_url=embed_url,
                embed_model_name=embed_model_name,
                embed_batch_size=1
            )
            
            # Создаем или получаем ретривер из кэша
            retriever = self._get_retriever(
                embedding=embedding,
                vector_store_manager=vector_store_manager,
                vdb_url=vdb_url
            )
            
            # Выполняем поиск без фильтрации
            results = retriever.retrieve(
                query=self.query,
                top_k=self.max_results,
                filter_metadata=None
            )
            
            # Форматируем результаты
            return self._format_results(results, self.query)
            
        except ServiceError as e:
            logger.error(f"ServiceError при выполнении RAG поиска: {e.detail}")
            return self._format_error(e.error, e.detail)
        except (ConnectionError, TimeoutError, OSError) as e:
            # Обработка ошибок подключения к Qdrant или эмбеддингам
            error_message = str(e)
            error_type = type(e).__name__
            
            # Проверяем ошибки подключения к Qdrant
            is_windows_connection_error = (
                isinstance(e, OSError) and hasattr(e, 'winerror') and e.winerror == 10061
            ) or "10061" in error_message
            
            if "timeout" in error_message.lower() or "Timeout" in error_type or isinstance(e, TimeoutError):
                logger.error(f"Таймаут при выполнении RAG поиска: {e}")
                return self._format_error(
                    "Таймаут подключения",
                    f"Таймаут подключения к сервису. Проверьте доступность Qdrant и сервиса эмбеддингов."
                )
            elif (
                "connection" in error_message.lower() or 
                "Connection" in error_type or 
                "connect" in error_message.lower() or
                is_windows_connection_error or
                isinstance(e, ConnectionError)
            ):
                logger.error(f"Ошибка подключения при выполнении RAG поиска: {e}")
                return self._format_error(
                    "Ошибка подключения",
                    f"Не удалось подключиться к сервису. Убедитесь, что Qdrant и сервис эмбеддингов запущены и доступны."
                )
            else:
                logger.exception(f"Ошибка подключения при выполнении RAG поиска: {e}")
                return self._format_error(
                    "Ошибка подключения",
                    f"Ошибка при подключении к сервису: {error_message}"
                )
        except Exception as e:
            logger.exception(f"Ошибка при выполнении RAG поиска: {e}")
            return self._format_error(
                "Ошибка при выполнении RAG поиска",
                str(e)
            )
    
    def _initialize_rag_components(
        self,
        vdb_url: str,
        embed_api_key: Optional[str] = None,
        embed_url: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        embed_batch_size: Optional[int] = None
    ) -> tuple:
        """
        Инициализация компонентов RAG (embedding, vector_store) с кэшированием.
        
        Returns:
            Tuple (embedding, vector_store_manager)
        """
        from rag.giga_embeddings import GigaEmbedding
        from rag.vector_store import QdrantVectorStoreManager
        from utils.config import get_config
        
        # Загрузка конфигурации для qdrant (кэшируется)
        if RAGTool._config_cache is None:
            RAGTool._config_cache = get_config()
        config = RAGTool._config_cache
        
        # Определяем параметры эмбеддингов: приоритет у параметров из запроса, затем значения по умолчанию
        if embed_api_key is not None and not embed_api_key.strip():
            raise ServiceError(
                error="Неверный API ключ для эмбеддингов",
                detail="API ключ указан в запросе, но является пустой строкой",
                code="empty_embed_api_key",
            )
        
        final_api_key = embed_api_key if embed_api_key and embed_api_key.strip() else os.getenv("GIGACHAT_AUTH_KEY")
        
        if not final_api_key:
            raise ServiceError(
                error="Не настроен API ключ для эмбеддингов",
                detail="API ключ не указан в запросе (embed_api_key) и переменная окружения GIGACHAT_AUTH_KEY не установлена",
                code="missing_embed_api_key",
            )
        
        final_api_url = embed_url or "https://gigachat.devices.sberbank.ru/api/v1"
        final_model = embed_model_name or "Embeddings"
        final_scope = "GIGACHAT_API_PERS"
        batch_size = embed_batch_size if embed_batch_size is not None else 10
        max_retries = 3
        timeout = 60
        
        # Создаем ключ кэша для эмбеддингов
        embedding_cache_key = f"{final_api_url}:{final_model}:{final_scope}:{batch_size}:{max_retries}:{timeout}"
        
        if embedding_cache_key not in RAGTool._embedding_cache:
            embedding = GigaEmbedding(
                credentials=final_api_key,
                scope=final_scope,
                api_url=final_api_url,
                model=final_model,
                batch_size=batch_size,
                max_retries=max_retries,
                timeout=timeout
            )
            RAGTool._embedding_cache[embedding_cache_key] = embedding
            logger.debug(f"Создан новый объект GigaEmbedding для {final_api_url}/{final_model} (кэширован)")
        
        embedding = RAGTool._embedding_cache[embedding_cache_key]
        
        # Инициализация векторного хранилища (кэшируется по vdb_url)
        qdrant_config = {}
        try:
            if isinstance(config, dict):
                qdrant_config = config.get("qdrant", {})
                if not isinstance(qdrant_config, dict):
                    qdrant_config = {}
            else:
                # Если config не словарь, пробуем через getattr и model_dump
                qdrant_section = getattr(config, "qdrant", None)
                if qdrant_section is not None:
                    try:
                        if hasattr(qdrant_section, "model_dump"):
                            qdrant_config = qdrant_section.model_dump()
                        elif isinstance(qdrant_section, dict):
                            qdrant_config = qdrant_section
                        else:
                            qdrant_config = {}
                    except Exception:
                        qdrant_config = {}
        except Exception:
            qdrant_config = {}
        
        # Убеждаемся, что qdrant_config это словарь
        if not isinstance(qdrant_config, dict):
            qdrant_config = {}
        normalized_url = vdb_url.strip().rstrip("/")
        if not normalized_url.startswith("http"):
            normalized_url = f"http://{normalized_url}"
        
        vector_store_cache_key = f"{normalized_url}:{qdrant_config.get('collection_name', 'smart_rag_documents')}:{qdrant_config.get('vector_size', 1024)}"
        
        if vector_store_cache_key not in RAGTool._vector_store_cache:
            vector_store_manager = QdrantVectorStoreManager(
                url=normalized_url,
                api_key=qdrant_config.get("api_key"),
                collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
            
            # Убеждаемся, что коллекция существует
            vector_store_manager.ensure_collection_exists()
            
            RAGTool._vector_store_cache[vector_store_cache_key] = vector_store_manager
            logger.debug(f"Создан новый объект QdrantVectorStoreManager для {normalized_url} (кэширован)")
        
        vector_store_manager = RAGTool._vector_store_cache[vector_store_cache_key]
        
        return embedding, vector_store_manager
    
    def _get_retriever(
        self,
        embedding,
        vector_store_manager,
        vdb_url: str
    ):
        """
        Получение или создание DocumentRetriever с кэшированием.
        
        Args:
            embedding: Объект эмбеддингов
            vector_store_manager: Менеджер векторного хранилища
            vdb_url: URL векторной БД (для ключа кэша)
        
        Returns:
            DocumentRetriever
        """
        from rag.retriever import DocumentRetriever
        from utils.config import get_config
        
        # Загрузка конфигурации
        if RAGTool._config_cache is None:
            RAGTool._config_cache = get_config()
        config = RAGTool._config_cache
        
        # Безопасное извлечение конфигурации RAG
        rag_config = {}
        try:
            if isinstance(config, dict):
                rag_config = config.get("rag", {})
                if not isinstance(rag_config, dict):
                    rag_config = {}
            else:
                # Если config не словарь, пробуем через getattr и model_dump
                rag_section = getattr(config, "rag", None)
                if rag_section is not None:
                    try:
                        if hasattr(rag_section, "model_dump"):
                            rag_config = rag_section.model_dump()
                        elif isinstance(rag_section, dict):
                            rag_config = rag_section
                        else:
                            rag_config = {}
                    except Exception:
                        rag_config = {}
        except Exception:
            rag_config = {}
        
        # Убеждаемся, что rag_config это словарь
        if not isinstance(rag_config, dict):
            rag_config = {}
        
        # Параметры ретривера из конфигурации
        top_k = rag_config.get("top_k", 5)
        
        hybrid_search_config = rag_config.get("hybrid_search", {})
        if not isinstance(hybrid_search_config, dict):
            hybrid_search_config = {}
        hybrid_search_enabled = hybrid_search_config.get("enabled", True)
        vector_top_k = hybrid_search_config.get("vector_top_k", 20)
        text_top_k = hybrid_search_config.get("text_top_k", 20)
        
        # Создаем ключ кэша для ретривера
        retriever_cache_key = f"{vdb_url}:{top_k}:{hybrid_search_enabled}:{vector_top_k}:{text_top_k}"
        
        # Инициализация реранкера (опционально)
        reranker = None
        reranker_config = rag_config.get("reranker", {})
        if not isinstance(reranker_config, dict):
            reranker_config = {}
        if reranker_config.get("enabled", False):
            try:
                from rag.reranker import ChatCompletionsReranker
                reranker = ChatCompletionsReranker(
                    model=reranker_config.get("model", "dengcao/Qwen3-Reranker-0.6B:F16"),
                    api_url=reranker_config.get("api_url", "http://localhost:11434"),
                    max_retries=reranker_config.get("max_retries", 3),
                    timeout=reranker_config.get("timeout", 60)
                )
                reranker_model = reranker_config.get('model', '')
                retriever_cache_key += f":reranker:{reranker_model}"
            except Exception as e:
                logger.warning(f"Не удалось инициализировать реранкер: {e}. Продолжаем без реранкера.")
        
        if retriever_cache_key not in RAGTool._retriever_cache:
            retriever = DocumentRetriever(
                embedding=embedding,
                vector_store_manager=vector_store_manager,
                top_k=top_k,
                hybrid_search_enabled=hybrid_search_enabled,
                vector_top_k=vector_top_k,
                text_top_k=text_top_k,
                reranker=reranker
            )
            RAGTool._retriever_cache[retriever_cache_key] = retriever
            logger.debug(f"Создан новый объект DocumentRetriever (кэширован)")
        
        return RAGTool._retriever_cache[retriever_cache_key]
    
    def _format_results(self, results: list[Dict[str, Any]], query: str) -> str:
        """
        Форматирование результатов поиска в читаемый текст.
        
        Args:
            results: Список результатов поиска
            query: Исходный запрос
        
        Returns:
            Отформатированная строка с результатами
        """
        if not results:
            return f"""RAG Search Results

Query: {query}
Results: No relevant documents found in the knowledge base.

The search did not return any results. This could mean:
- The query doesn't match any indexed documents
- The documents haven't been indexed yet
- Try rephrasing your query or checking if documents are indexed"""
        
        formatted_parts = [
            f"RAG Search Results",
            f"",
            f"Query: {query}",
            f"Found {len(results)} relevant document(s):",
            f""
        ]
        
        for idx, result in enumerate(results, 1):
            text = result.get("text", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # Извлекаем метаданные
            file_name = metadata.get("file_name", "Unknown")
            irv_id = metadata.get("irv_id", "Unknown")
            irvf_id = metadata.get("irvf_id", "")
            chunk_type = metadata.get("chunk_type", "text")
            
            # Формируем информацию о документе
            doc_info = f"[{idx}] Document: {file_name}"
            if irv_id != "Unknown":
                doc_info += f" (IRV ID: {irv_id})"
            if irvf_id:
                doc_info += f" (File ID: {irvf_id})"
            if chunk_type != "text":
                doc_info += f" (Type: {chunk_type})"
            
            formatted_parts.append(doc_info)
            formatted_parts.append(f"Relevance Score: {score:.4f}")
            formatted_parts.append(f"Content:")
            formatted_parts.append(f"{text[:500]}{'...' if len(text) > 500 else ''}")
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def _format_error(self, error: str, detail: str) -> str:
        """
        Форматирование ошибки в читаемый текст.
        
        Args:
            error: Краткое описание ошибки
            detail: Детальное описание ошибки
        
        Returns:
            Отформатированная строка с ошибкой
        """
        return f"""RAG Search Error

Error: {error}
Detail: {detail}

Please check:
- VDB URL is correct and Qdrant server is running
- Embedding API key is configured
- Documents are indexed in the vector database
- Network connectivity to Qdrant and embedding service"""
