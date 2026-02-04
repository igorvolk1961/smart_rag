"""
Сервис для работы с RAG (Retrieval-Augmented Generation).
"""

import base64
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList, PointStruct

from api.exceptions import ServiceError
from api.models.llm_models import (
    RAGAddResponse,
    RAGRemoveResponse,
    RAGRequest,
    CollectionListRequest,
    CollectionListResponse,
    CollectionInfo,
    CollectionDeleteRequest,
    CollectionDeleteResponse,
)
from api.siu_client import SiuClient


class RAGService:
    """Сервис для управления файлами в RAG-системе."""

    def add_files_to_rag(self, request: RAGRequest, siu_client: SiuClient) -> Dict[str, Any]:
        """
        Добавление файлов ИО в векторную БД.
        
        Args:
            request: Запрос с параметрами
            siu_client: Клиент для работы с СИУ
            
        Returns:
            Словарь с результатами добавления файлов
        """
        try:
            # Получение полной информации об ИО с метаданными и файлами
            irv_info = siu_client.get_irv(
                request.irv_id,
                with_meta=True,
                with_base_metas=True,
                with_files=True
            )
            
            # Извлечение метаданных ИО
            irv_metadata = self._extract_irv_metadata(irv_info, request.irv_id)
            
            # Получение и фильтрация файлов
            files_to_process = self._get_files_to_process(siu_client, request.irv_id)
            
            if not files_to_process:
                return RAGAddResponse(
                    success=True,
                    irv_id=request.irv_id,
                    files_processed=0,
                    chunks_saved=0,
                    toc_chunks_saved=0,
                    files_info=[]
                ).model_dump()
            
            # Создание временной директории для файлов
            temp_dir = tempfile.mkdtemp(prefix="rag_processing_")
            temp_path = Path(temp_dir)
            chunker_temp_dir = tempfile.mkdtemp(prefix="rag_chunks_")
            
            try:
                # Инициализация компонентов RAG
                chunker, embedding, vector_store_manager = self._initialize_rag_components(
                    request, 
                    chunker_output_dir=str(Path(chunker_temp_dir) / "chunks")
                )
                
                # Обработка файлов
                total_chunks = 0
                total_toc_chunks = 0
                files_info = []
                
                for file_data in files_to_process:
                    file_result = self._process_file(
                        file_data,
                        request.irv_id,
                        irv_metadata,
                        siu_client,
                        chunker,
                        embedding,
                        vector_store_manager,
                        temp_path
                    )
                    if file_result:
                        total_chunks += file_result["chunks_count"]
                        total_toc_chunks += file_result["toc_chunks_count"]
                        files_info.append(file_result)
                
                return RAGAddResponse(
                    success=True,
                    irv_id=request.irv_id,
                    files_processed=len(files_to_process),
                    chunks_saved=total_chunks,
                    toc_chunks_saved=total_toc_chunks,
                    files_info=files_info
                ).model_dump()
                
            finally:
                # Очистка временных директорий
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
                if Path(chunker_temp_dir).exists():
                    shutil.rmtree(chunker_temp_dir, ignore_errors=True)
                    
        except Exception as e:
            logger.exception(f"Ошибка при добавлении файлов в RAG: {e}")
            raise

    def remove_files_from_rag(self, request: RAGRequest) -> Dict[str, Any]:
        """
        Удаление файлов ИО из векторной БД.
        
        Args:
            request: Запрос с параметрами
            
        Returns:
            Словарь с результатами удаления файлов
        """
        try:
            from rag.vector_store import QdrantVectorStoreManager
            from utils.config import get_config
            
            # Загрузка конфигурации
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            
            # Инициализация векторного хранилища
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            
            vector_store_manager = QdrantVectorStoreManager(
                url=vdb_url,
                api_key=qdrant_config.get("api_key"),
                collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
            
            # Создание фильтра для поиска точек с указанным irv_id
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="irv_id",
                        match=MatchValue(value=request.irv_id)
                    )
                ]
            )
            
            # Получаем список точек для удаления через scroll
            scroll_result = vector_store_manager.client.scroll(
                collection_name=vector_store_manager.collection_name,
                scroll_filter=filter_condition,
                limit=10000,  # Максимальное количество точек для удаления за раз
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = [point.id for point in scroll_result[0]]
            
            if not points_to_delete:
                return RAGRemoveResponse(
                    success=True,
                    irv_id=request.irv_id,
                    chunks_deleted=0
                ).model_dump()
            
            # Удаление точек батчами
            batch_size = 1000
            deleted_count = 0
            for i in range(0, len(points_to_delete), batch_size):
                batch = points_to_delete[i:i + batch_size]
                vector_store_manager.client.delete(
                    collection_name=vector_store_manager.collection_name,
                    points_selector=PointIdsList(points=batch)
                )
                deleted_count += len(batch)
            
            logger.info(f"Удалено {deleted_count} чанков для irv_id={request.irv_id}")
            
            return RAGRemoveResponse(
                success=True,
                irv_id=request.irv_id,
                chunks_deleted=deleted_count
            ).model_dump()
            
        except Exception as e:
            logger.exception(f"Ошибка при удалении файлов из RAG: {e}")
            raise

    def _extract_irv_metadata(self, irv_info: Any, irv_id: str) -> Dict[str, Any]:
        """Извлечение метаданных ИО из ответа get_irv."""
        irv_metadata = {}
        if isinstance(irv_info, dict):
            # Извлекаем основные поля ИО для сохранения в метаданных чанков
            irv_metadata = {
                "irv_id": irv_id,
                "irv_name": irv_info.get("name"),
                "irv_description": irv_info.get("description"),
            }
            
            # Добавляем метаданные, если они есть
            if "meta" in irv_info and isinstance(irv_info["meta"], dict):
                irv_metadata["irv_meta"] = irv_info["meta"]
            elif "metadata" in irv_info and isinstance(irv_info["metadata"], dict):
                irv_metadata["irv_metadata"] = irv_info["metadata"]
            elif "data" in irv_info and isinstance(irv_info["data"], dict):
                meta_data = irv_info["data"].get("meta", {})
                if meta_data:
                    irv_metadata["irv_meta"] = meta_data
            
            # Удаляем None значения
            irv_metadata = {k: v for k, v in irv_metadata.items() if v is not None}
        
        return irv_metadata

    def _get_files_to_process(self, siu_client: SiuClient, irv_id: str) -> List[Dict[str, Any]]:
        """Получение и фильтрация файлов для обработки."""
        # Получение списка файлов
        raw_files = siu_client.get_irv_files(irv_id)
        
        # Нормализация списка файлов (аналогично chat_history.py)
        if isinstance(raw_files, list):
            files_list = raw_files
        elif isinstance(raw_files, dict):
            files_list = raw_files.get("contents", [])
            if not isinstance(files_list, list):
                files_list = [raw_files] if raw_files else []
        else:
            files_list = []
        
        # Фильтрация файлов по расширению (docx и txt)
        supported_extensions = {".docx", ".txt"}
        files_to_process = []
        for file_item in files_list:
            # Нормализация элемента файла
            file_data = file_item.get("data", file_item) if isinstance(file_item, dict) else file_item
            if not isinstance(file_data, dict):
                continue
            
            # Извлечение имени файла и irvfId
            file_name = file_data.get("name") or file_data.get("fileName") or ""
            irvf_id = file_data.get("irvfId") or file_data.get("id")
            
            if not file_name or not irvf_id:
                logger.warning(f"Пропущен файл: отсутствует name или irvfId")
                continue
            
            file_ext = Path(file_name).suffix.lower()
            if file_ext in supported_extensions:
                files_to_process.append({
                    "name": file_name,
                    "irvfId": irvf_id,
                    **file_data  # Сохраняем все остальные поля
                })
        
        return files_to_process

    def _initialize_rag_components(self, request: RAGRequest, chunker_output_dir: str):
        """Инициализация компонентов RAG (chunker, embedding, vector_store)."""
        from rag.chunker_integration import ChunkerIntegration
        from rag.giga_embeddings import GigaEmbedding
        from rag.vector_store import QdrantVectorStoreManager
        from utils.config import get_config
        
        # Загрузка конфигурации
        config = get_config()
        
        # Инициализация компонентов RAG
        chunker_config = config.get("chunker", {})
        qdrant_config = config.get("qdrant", {})
        embeddings_config = config.get("embeddings", {}).get("giga", {})
        
        # Инициализация chunker
        chunker = ChunkerIntegration(
            chunker_config_path=chunker_config.get("config_path", "smartchanker_config.json"),
            output_dir=chunker_output_dir
        )
        
        # Инициализация эмбеддингов
        embed_api_key = os.getenv("GIGACHAT_AUTH_KEY")
        if not embed_api_key:
            raise ServiceError(
                error="Не настроен API ключ для эмбеддингов",
                detail="Переменная окружения GIGACHAT_AUTH_KEY не установлена",
                code="missing_embed_api_key",
            )
        
        embedding = GigaEmbedding(
            auth_key=embed_api_key,
            scope=embeddings_config.get("scope", "GIGACHAT_API_PERS"),
            api_url=embeddings_config.get("api_url", "https://gigachat.devices.sberbank.ru/api/v1"),
            model=embeddings_config.get("model", "Embeddings"),
            batch_size=embeddings_config.get("batch_size", 10),
            max_retries=embeddings_config.get("max_retries", 3),
            timeout=embeddings_config.get("timeout", 60)
        )
        
        # Инициализация векторного хранилища
        vdb_url = request.vdb_url.strip().rstrip("/")
        if not vdb_url.startswith("http"):
            vdb_url = f"http://{vdb_url}"
        
        vector_store_manager = QdrantVectorStoreManager(
            url=vdb_url,
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
            vector_size=qdrant_config.get("vector_size", 1024),
            timeout=qdrant_config.get("timeout", 30)
        )
        
        # Убеждаемся, что коллекция существует
        vector_store_manager.ensure_collection_exists()
        
        return chunker, embedding, vector_store_manager

    def _process_file(
        self,
        file_data: Dict[str, Any],
        irv_id: str,
        irv_metadata: Dict[str, Any],
        siu_client: SiuClient,
        chunker,
        embedding,
        vector_store_manager,
        temp_path: Path
    ) -> Dict[str, Any]:
        """Обработка одного файла."""
        file_name = file_data.get("name", "unknown")
        irvf_id = file_data.get("irvfId")
        
        if not irvf_id:
            logger.warning(f"Пропущен файл {file_name}: отсутствует irvfId")
            return None
        
        try:
            # Получение содержимого файла
            file_content_response = siu_client.get_irv_file_content(file_data)
            
            # Определение формата содержимого
            file_bytes = self._extract_file_content(file_content_response, file_name)
            if file_bytes is None:
                return {
                    "file_name": file_name,
                    "irvf_id": irvf_id,
                    "status": "error",
                    "error": "Не удалось извлечь содержимое файла",
                    "chunks_count": 0,
                    "toc_chunks_count": 0
                }
            
            # Сохранение файла во временную директорию
            temp_file_path = temp_path / file_name
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            
            # Обработка документа через chunker
            chunker_result = chunker.process_document(
                str(temp_file_path),
                document_id=f"{irv_id}_{irvf_id}"
            )
            
            # Подготовка метаданных для сохранения
            doc_metadata = {
                "irv_id": irv_id,
                "irvf_id": irvf_id,
                "file_name": file_name,
                **irv_metadata
            }
            
            # Создание узлов из чанков
            nodes, toc_nodes = self._create_nodes_from_chunks(
                chunker_result,
                doc_metadata,
                irv_id,
                irvf_id
            )
            
            # Индексация узлов
            if nodes or toc_nodes:
                all_nodes = nodes + toc_nodes
                texts = [node.text for node in all_nodes]
                embeddings_list = embedding._get_text_embeddings(texts)
                
                if len(embeddings_list) == len(all_nodes):
                    # Сохранение в Qdrant
                    points = []
                    for node, emb in zip(all_nodes, embeddings_list):
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=emb,
                            payload={
                                "text": node.text,
                                **node.metadata
                            }
                        )
                        points.append(point)
                    
                    vector_store_manager.client.upsert(
                        collection_name=vector_store_manager.collection_name,
                        points=points
                    )
                    
                    return {
                        "file_name": file_name,
                        "irvf_id": irvf_id,
                        "chunks_count": len(nodes),
                        "toc_chunks_count": len(toc_nodes),
                        "status": "success"
                    }
                else:
                    logger.error(f"Несоответствие количества эмбеддингов для файла {file_name}")
                    return {
                        "file_name": file_name,
                        "irvf_id": irvf_id,
                        "status": "error",
                        "error": "Несоответствие количества эмбеддингов",
                        "chunks_count": 0,
                        "toc_chunks_count": 0
                    }
            else:
                logger.warning(f"Не найдено чанков для файла {file_name}")
                return {
                    "file_name": file_name,
                    "irvf_id": irvf_id,
                    "chunks_count": 0,
                    "toc_chunks_count": 0,
                    "status": "no_chunks"
                }
            
        except Exception as e:
            logger.exception(f"Ошибка при обработке файла {file_name}: {e}")
            return {
                "file_name": file_name,
                "irvf_id": irvf_id,
                "status": "error",
                "error": str(e),
                "chunks_count": 0,
                "toc_chunks_count": 0
            }
        finally:
            # Удаление временного файла
            temp_file_path = temp_path / file_name
            if temp_file_path.exists():
                temp_file_path.unlink()

    def _extract_file_content(self, file_content_response: Any, file_name: str) -> bytes:
        """Извлечение содержимого файла из ответа СИУ."""
        if file_content_response is None:
            logger.warning(f"Содержимое файла {file_name} пусто")
            return None
        
        if isinstance(file_content_response, (bytes, bytearray)):
            return bytes(file_content_response)
        elif isinstance(file_content_response, dict):
            # Если ответ содержит base64 или другой формат
            content_data = file_content_response.get("data") or file_content_response.get("content")
            if content_data is None:
                logger.warning(f"Не найдено содержимое файла {file_name} в ответе")
                return None
            if isinstance(content_data, (bytes, bytearray)):
                return bytes(content_data)
            elif isinstance(content_data, str):
                # Пробуем декодировать base64
                try:
                    return base64.b64decode(content_data)
                except Exception:
                    # Если не base64, то это просто текст
                    return content_data.encode("utf-8")
            else:
                return str(content_data).encode("utf-8")
        else:
            # Если это строка или другой тип
            if isinstance(file_content_response, str):
                return file_content_response.encode("utf-8")
            else:
                return str(file_content_response).encode("utf-8")

    def _create_nodes_from_chunks(
        self,
        chunker_result: Dict[str, Any],
        doc_metadata: Dict[str, Any],
        irv_id: str,
        irvf_id: str
    ):
        """Создание узлов LlamaIndex из чанков SmartChanker."""
        from llama_index.core.schema import TextNode
        
        nodes = []
        toc_nodes = []
        
        # Обработка обычных чанков
        for idx, chunk_data in enumerate(chunker_result.get("chunks", [])):
            text = chunk_data.get("text", "")
            if not text or not text.strip():
                continue
            
            chunk_metadata = chunk_data.get("metadata", {})
            node_metadata = {
                **doc_metadata,
                **chunk_metadata,
                "chunk_index": idx,
                "chunk_type": "text"
            }
            
            node = TextNode(
                text=text.strip(),
                metadata=node_metadata,
                id_=f"{irv_id}_{irvf_id}_chunk_{idx}"
            )
            nodes.append(node)
        
        # Обработка чанков оглавления
        for idx, toc_chunk_data in enumerate(chunker_result.get("toc_chunks", [])):
            text = toc_chunk_data.get("text", "")
            if not text or not text.strip():
                continue
            
            chunk_metadata = toc_chunk_data.get("metadata", {})
            node_metadata = {
                **doc_metadata,
                **chunk_metadata,
                "chunk_index": idx,
                "chunk_type": "toc"
            }
            
            node = TextNode(
                text=text.strip(),
                metadata=node_metadata,
                id_=f"{irv_id}_{irvf_id}_toc_{idx}"
            )
            toc_nodes.append(node)
        
        return nodes, toc_nodes

    def get_collections(self, request: CollectionListRequest) -> Dict[str, Any]:
        """
        Получение списка коллекций в векторной БД.
        
        Args:
            request: Запрос с параметрами vdb_url
            
        Returns:
            Словарь со списком коллекций
        """
        try:
            from rag.vector_store import QdrantVectorStoreManager
            from utils.config import get_config
            
            # Загрузка конфигурации
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            
            # Инициализация векторного хранилища
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            
            # Создаем временный менеджер для подключения к Qdrant
            vector_store_manager = QdrantVectorStoreManager(
                url=vdb_url,
                api_key=qdrant_config.get("api_key"),
                collection_name="temp",  # Временное имя, не используется
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
            
            # Получаем список коллекций
            collections_response = vector_store_manager.client.get_collections()
            collections_list = collections_response.collections if hasattr(collections_response, 'collections') else []
            
            # Формируем информацию о каждой коллекции
            collections_info = []
            for collection in collections_list:
                collection_name = collection.name if hasattr(collection, 'name') else str(collection)
                
                # Получаем детальную информацию о коллекции
                try:
                    collection_detail = vector_store_manager.client.get_collection(collection_name)
                    
                    collection_data = {
                        "name": collection_name,
                        "points_count": getattr(collection_detail, 'points_count', 0),
                    }
                    
                    # Статус коллекции
                    if hasattr(collection_detail, 'status'):
                        status = collection_detail.status
                        if hasattr(status, 'name'):
                            collection_data["status"] = status.name
                        elif isinstance(status, str):
                            collection_data["status"] = status
                        else:
                            collection_data["status"] = str(status)
                    
                    # Конфигурация векторов
                    if hasattr(collection_detail, 'config'):
                        config_obj = collection_detail.config
                        if hasattr(config_obj, 'params') and hasattr(config_obj.params, 'vectors'):
                            vectors_config = config_obj.params.vectors
                            
                            if hasattr(vectors_config, 'size'):
                                collection_data["vector_size"] = vectors_config.size
                            
                            if hasattr(vectors_config, 'distance'):
                                distance = vectors_config.distance
                                if hasattr(distance, 'name'):
                                    collection_data["distance"] = distance.name
                                elif isinstance(distance, str):
                                    collection_data["distance"] = distance
                                else:
                                    collection_data["distance"] = str(distance)
                    
                    collections_info.append(CollectionInfo(**collection_data))
                    
                except Exception as e:
                    logger.warning(f"Не удалось получить детальную информацию о коллекции {collection_name}: {e}")
                    # Добавляем базовую информацию
                    collections_info.append(CollectionInfo(
                        name=collection_name,
                        points_count=0,
                        status=None,
                        vector_size=None,
                        distance=None
                    ))
            
            return CollectionListResponse(
                success=True,
                collections=collections_info,
                total=len(collections_info)
            ).model_dump()
            
        except Exception as e:
            logger.exception(f"Ошибка при получении списка коллекций: {e}")
            raise

    def delete_collection(self, request: CollectionDeleteRequest) -> Dict[str, Any]:
        """
        Удаление коллекции из векторной БД.
        
        Args:
            request: Запрос с параметрами vdb_url и collection_name
            
        Returns:
            Словарь с результатом удаления коллекции
        """
        try:
            from rag.vector_store import QdrantVectorStoreManager
            from utils.config import get_config
            
            # Загрузка конфигурации
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            
            # Инициализация векторного хранилища
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            
            # Создаем менеджер с указанной коллекцией для удаления
            vector_store_manager = QdrantVectorStoreManager(
                url=vdb_url,
                api_key=qdrant_config.get("api_key"),
                collection_name=request.collection_name,
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
            
            # Проверяем существование коллекции перед удалением
            try:
                collections_response = vector_store_manager.client.get_collections()
                collections_list = collections_response.collections if hasattr(collections_response, 'collections') else []
                collection_exists = any(
                    col.name == request.collection_name for col in collections_list
                )
                
                if not collection_exists:
                    return CollectionDeleteResponse(
                        success=False,
                        collection_name=request.collection_name,
                        message=f"Коллекция '{request.collection_name}' не найдена"
                    ).model_dump()
            except Exception as e:
                logger.warning(f"Не удалось проверить существование коллекции: {e}")
            
            # Удаляем коллекцию
            vector_store_manager.delete_collection()
            
            logger.info(f"Коллекция {request.collection_name} успешно удалена")
            
            return CollectionDeleteResponse(
                success=True,
                collection_name=request.collection_name,
                message=f"Коллекция '{request.collection_name}' успешно удалена"
            ).model_dump()
            
        except Exception as e:
            logger.exception(f"Ошибка при удалении коллекции: {e}")
            raise
