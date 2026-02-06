"""
Сервис для работы с RAG (Retrieval-Augmented Generation).
"""

import base64
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList, PointStruct

from api.exceptions import ServiceError
from api.models.llm_models import (
    RAGAddResponse,
    RAGRemoveResponse,
    RAGInfoResponse,
    RAGRequest,
    CollectionListRequest,
    CollectionListResponse,
    CollectionInfo,
    CollectionDeleteRequest,
    CollectionDeleteResponse,
)
from api.siu_client import SiuClient


class RAGService:
    """Сервис для работы с RAG."""
    
    @staticmethod
    def _handle_qdrant_connection_error(e: Exception, vdb_url: str) -> ServiceError:
        """
        Обработка ошибок подключения к Qdrant с понятным сообщением.
        
        Args:
            e: Исключение, возникшее при работе с Qdrant
            vdb_url: URL Qdrant сервера
            
        Returns:
            ServiceError с понятным сообщением
        """
        error_message = str(e)
        error_type = type(e).__name__
        
        # Проверяем различные типы ошибок подключения
        # Windows ошибка 10061: "Подключение не установлено, т.к. конечный компьютер отверг запрос на подключение"
        is_windows_connection_error = (
            isinstance(e, OSError) and hasattr(e, 'winerror') and e.winerror == 10061
        ) or "10061" in error_message
        
        if "timeout" in error_message.lower() or "Timeout" in error_type or isinstance(e, TimeoutError):
            logger.error(f"Таймаут подключения к Qdrant на {vdb_url}: {e}")
            return ServiceError(
                error="Qdrant недоступен",
                detail=f"Таймаут подключения к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен и доступен.",
                code="qdrant_timeout"
            )
        elif (
            "connection" in error_message.lower() or 
            "Connection" in error_type or 
            "connect" in error_message.lower() or
            is_windows_connection_error or
            isinstance(e, (ConnectionError, ConnectionRefusedError))
        ):
            logger.error(f"Ошибка подключения к Qdrant на {vdb_url}: {e}")
            return ServiceError(
                error="Qdrant недоступен",
                detail=f"Не удалось подключиться к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен и доступен.",
                code="qdrant_connection_error"
            )
        else:
            logger.exception(f"Ошибка при работе с Qdrant на {vdb_url}: {e}")
            return ServiceError(
                error="Ошибка при работе с Qdrant",
                detail=f"Ошибка при работе с Qdrant сервером на {vdb_url}: {error_message}",
                code="qdrant_error"
            )
    """Сервис для управления файлами в RAG-системе."""
    
    # Кэш для переиспользуемых компонентов
    _embedding_cache: Dict[str, Any] = {}
    _vector_store_cache: Dict[str, Any] = {}
    _config_cache: Any = None

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
                    table_chunks_saved=0,
                    files_info=[]
                ).model_dump()
            
            # Создание временной директории для файлов
            temp_dir = tempfile.mkdtemp(prefix="rag_processing_")
            temp_path = Path(temp_dir)
            chunker_temp_dir = tempfile.mkdtemp(prefix="rag_chunks_")
            
            try:
                # Инициализация компонентов RAG
                try:
                    chunker, embedding, vector_store_manager = self._initialize_rag_components(
                        request, 
                        chunker_output_dir=str(Path(chunker_temp_dir) / "chunks")
                    )
                except Exception as init_error:
                    # Обрабатываем ошибки подключения к Qdrant при инициализации
                    vdb_url = request.vdb_url.strip().rstrip("/")
                    if not vdb_url.startswith("http"):
                        vdb_url = f"http://{vdb_url}"
                    raise self._handle_qdrant_connection_error(init_error, vdb_url)
                
                # Удаляем все старые чанки с этим irv_id перед добавлением новых
                # Это предотвращает создание дубликатов при повторном сохранении документа
                vdb_url = request.vdb_url.strip().rstrip("/")
                if not vdb_url.startswith("http"):
                    vdb_url = f"http://{vdb_url}"
                try:
                    deleted_count = self._delete_chunks_by_irv_id(
                        vector_store_manager,
                        request.irv_id,
                        vdb_url
                    )
                    if deleted_count > 0:
                        logger.info(f"Перед добавлением новых чанков удалено {deleted_count} старых чанков для irv_id={request.irv_id}")
                except ServiceError:
                    # Пробрасываем ServiceError (ошибки подключения к Qdrant)
                    raise
                except Exception as e:
                    # Обрабатываем неожиданные ошибки при удалении
                    logger.warning(f"Ошибка при удалении старых чанков для irv_id={request.irv_id}: {e}. Продолжаем добавление новых чанков.")
                    # Не прерываем выполнение - продолжаем добавление новых чанков
                
                # Обработка файлов
                total_chunks = 0
                total_toc_chunks = 0
                total_table_chunks = 0
                files_info = []
                errors = []
                
                for file_data in files_to_process:
                    file_result = self._process_file(
                        file_data,
                        request.irv_id,
                        irv_metadata,
                        siu_client,
                        chunker,
                        embedding,
                        vector_store_manager,
                        temp_path,
                        max_chunk_size=getattr(request, 'max_chunk_size', None)
                    )
                    if file_result:
                        # Проверяем статус обработки файла
                        if file_result.get("status") == "error":
                            error_msg = file_result.get("error", "Неизвестная ошибка")
                            file_name = file_result.get("file_name", "неизвестный файл")
                            errors.append(f"{file_name}: {error_msg}")
                            logger.error(f"Ошибка при обработке файла {file_name}: {error_msg}")
                        else:
                            # Учитываем чанки только для успешно обработанных файлов
                            total_chunks += file_result.get("chunks_count", 0)
                            total_toc_chunks += file_result.get("toc_chunks_count", 0)
                            total_table_chunks += file_result.get("table_chunks_count", 0)
                        files_info.append(file_result)
                
                # Если были ошибки, выбрасываем исключение с деталями
                if errors:
                    error_details = "; ".join(errors)
                    raise ServiceError(
                        error="Ошибка при обработке файлов",
                        detail=f"При обработке файлов возникли ошибки: {error_details}",
                        code="rag_processing_error"
                    )
                
                return RAGAddResponse(
                    success=True,
                    irv_id=request.irv_id,
                    files_processed=len(files_to_process),
                    chunks_saved=total_chunks,
                    toc_chunks_saved=total_toc_chunks,
                    table_chunks_saved=total_table_chunks,
                    files_info=files_info
                ).model_dump()
                
            finally:
                # Очистка временных директорий
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
                if Path(chunker_temp_dir).exists():
                    shutil.rmtree(chunker_temp_dir, ignore_errors=True)
                    
        except ServiceError:
            # Пробрасываем ServiceError как есть (уже обработанные ошибки)
            raise
        except Exception as e:
            # Обрабатываем необработанные ошибки подключения к Qdrant
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            raise self._handle_qdrant_connection_error(e, vdb_url)

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
            
            try:
                vector_store_manager = QdrantVectorStoreManager(
                    url=vdb_url,
                    api_key=qdrant_config.get("api_key"),
                    collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
                    vector_size=qdrant_config.get("vector_size", 1024),
                    timeout=qdrant_config.get("timeout", 30)
                )
            except Exception as init_error:
                # Обрабатываем ошибки подключения к Qdrant при инициализации
                raise self._handle_qdrant_connection_error(init_error, vdb_url)
            
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
            try:
                scroll_result = vector_store_manager.client.scroll(
                    collection_name=vector_store_manager.collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,  # Максимальное количество точек для удаления за раз
                    with_payload=False,
                    with_vectors=False
                )
            except Exception as scroll_error:
                # Обрабатываем ошибки подключения к Qdrant при выполнении операций
                raise self._handle_qdrant_connection_error(scroll_error, vdb_url)
            
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
                try:
                    vector_store_manager.client.delete(
                        collection_name=vector_store_manager.collection_name,
                        points_selector=PointIdsList(points=batch)
                    )
                    deleted_count += len(batch)
                except Exception as delete_error:
                    # Обрабатываем ошибки подключения к Qdrant при удалении
                    raise self._handle_qdrant_connection_error(delete_error, vdb_url)
            
            logger.info(f"Удалено {deleted_count} чанков для irv_id={request.irv_id}")
            
            return RAGRemoveResponse(
                success=True,
                irv_id=request.irv_id,
                chunks_deleted=deleted_count
            ).model_dump()
            
        except ServiceError:
            # Пробрасываем ServiceError как есть (уже обработанные ошибки)
            raise
        except Exception as e:
            # Обрабатываем необработанные ошибки подключения к Qdrant
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            raise self._handle_qdrant_connection_error(e, vdb_url)

    def get_file_info(self, request: RAGRequest) -> Dict[str, Any]:
        """
        Получение информации о файле в векторной БД.
        
        Args:
            request: Запрос с параметрами irv_id и vdb_url
            
        Returns:
            Словарь с информацией о файле
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
            
            try:
                vector_store_manager = QdrantVectorStoreManager(
                    url=vdb_url,
                    api_key=qdrant_config.get("api_key"),
                    collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
                    vector_size=qdrant_config.get("vector_size", 1024),
                    timeout=qdrant_config.get("timeout", 30)
                )
            except Exception as init_error:
                # Обрабатываем ошибки подключения к Qdrant при инициализации
                raise self._handle_qdrant_connection_error(init_error, vdb_url)
            
            # Создание фильтра для поиска точек с указанным irv_id
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="irv_id",
                        match=MatchValue(value=request.irv_id)
                    )
                ]
            )
            
            # Получаем все точки с payload для анализа
            try:
                scroll_result = vector_store_manager.client.scroll(
                    collection_name=vector_store_manager.collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,  # Максимальное количество точек для анализа
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as scroll_error:
                # Обрабатываем ошибки подключения к Qdrant при выполнении операций
                raise self._handle_qdrant_connection_error(scroll_error, vdb_url)
            
            points = scroll_result[0]
            
            if not points:
                # Файл не найден в векторной БД
                return RAGInfoResponse(
                    success=True,
                    irv_id=request.irv_id,
                    total_chunks=0,
                    text_chunks=0,
                    toc_chunks=0,
                    table_chunks=0,
                    files_info=[]
                ).model_dump()
            
            # Анализируем точки для подсчета статистики
            text_chunks = 0
            toc_chunks = 0
            table_chunks = 0
            files_dict = {}  # Словарь для группировки по файлам
            
            for point in points:
                payload = point.payload if hasattr(point, 'payload') else {}
                
                # Определяем тип чанка
                chunk_type = payload.get("chunk_type", "text")
                is_toc = payload.get("is_toc", False)
                is_table = payload.get("is_table", False)
                
                if is_table or chunk_type == "table":
                    table_chunks += 1
                elif is_toc or chunk_type == "toc":
                    toc_chunks += 1
                else:
                    text_chunks += 1
                
                # Группируем по файлам
                file_name = payload.get("file_name", "unknown")
                irvf_id = payload.get("irvf_id", "")
                file_key = f"{file_name}_{irvf_id}"
                
                if file_key not in files_dict:
                    files_dict[file_key] = {
                        "file_name": file_name,
                        "irvf_id": irvf_id,
                        "text_chunks": 0,
                        "toc_chunks": 0,
                        "table_chunks": 0,
                        "total_chunks": 0
                    }
                
                files_dict[file_key]["total_chunks"] += 1
                if is_table or chunk_type == "table":
                    files_dict[file_key]["table_chunks"] += 1
                elif is_toc or chunk_type == "toc":
                    files_dict[file_key]["toc_chunks"] += 1
                else:
                    files_dict[file_key]["text_chunks"] += 1
            
            # Преобразуем словарь в список
            files_info = list(files_dict.values())
            
            return RAGInfoResponse(
                success=True,
                irv_id=request.irv_id,
                total_chunks=len(points),
                text_chunks=text_chunks,
                toc_chunks=toc_chunks,
                table_chunks=table_chunks,
                files_info=files_info
            ).model_dump()
            
        except ServiceError:
            # Пробрасываем ServiceError как есть (уже обработанные ошибки)
            raise
        except Exception as e:
            # Обрабатываем необработанные ошибки подключения к Qdrant
            vdb_url = request.vdb_url.strip().rstrip("/")
            if not vdb_url.startswith("http"):
                vdb_url = f"http://{vdb_url}"
            raise self._handle_qdrant_connection_error(e, vdb_url)

    def _delete_chunks_by_irv_id(
        self, 
        vector_store_manager, 
        irv_id: str,
        vdb_url: str
    ) -> int:
        """
        Удаление всех чанков с указанным irv_id из векторной БД.
        
        Args:
            vector_store_manager: Менеджер векторного хранилища Qdrant
            irv_id: Идентификатор версии информационного объекта
            vdb_url: URL векторной БД (для обработки ошибок)
            
        Returns:
            Количество удаленных чанков
            
        Raises:
            ServiceError: При ошибках подключения к Qdrant
        """
        try:
            # Создание фильтра для поиска точек с указанным irv_id
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="irv_id",
                        match=MatchValue(value=irv_id)
                    )
                ]
            )
            
            # Получаем список точек для удаления через scroll
            try:
                scroll_result = vector_store_manager.client.scroll(
                    collection_name=vector_store_manager.collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,  # Максимальное количество точек для удаления за раз
                    with_payload=False,
                    with_vectors=False
                )
            except Exception as scroll_error:
                # Обрабатываем ошибки подключения к Qdrant при выполнении операций
                raise self._handle_qdrant_connection_error(scroll_error, vdb_url)
            
            points_to_delete = [point.id for point in scroll_result[0]]
            
            if not points_to_delete:
                logger.debug(f"Не найдено чанков для удаления с irv_id={irv_id}")
                return 0
            
            # Удаление точек батчами
            batch_size = 1000
            deleted_count = 0
            for i in range(0, len(points_to_delete), batch_size):
                batch = points_to_delete[i:i + batch_size]
                try:
                    vector_store_manager.client.delete(
                        collection_name=vector_store_manager.collection_name,
                        points_selector=PointIdsList(points=batch)
                    )
                    deleted_count += len(batch)
                except Exception as delete_error:
                    # Обрабатываем ошибки подключения к Qdrant при удалении
                    raise self._handle_qdrant_connection_error(delete_error, vdb_url)
            
            logger.info(f"Удалено {deleted_count} старых чанков для irv_id={irv_id}")
            return deleted_count
            
        except ServiceError:
            # Пробрасываем ServiceError как есть (уже обработанные ошибки)
            raise
        except Exception as e:
            # Обрабатываем необработанные ошибки подключения к Qdrant
            raise self._handle_qdrant_connection_error(e, vdb_url)

    def _filter_attr_map_metadata(self, attr_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Фильтрация метаданных из attrMap по заданным правилам.
        
        Args:
            attr_map: Словарь метаданных, где ключ - название, значение - словарь с полями meta и value
            
        Returns:
            Отфильтрованный словарь метаданных
        """
        filtered_metadata = {}
        
        if not isinstance(attr_map, dict):
            return filtered_metadata
        
        # Исключаемые названия метаданных
        excluded_names = {
            "Настройка доступа к базе знаний",
            "Действие в базе знаний"
        }
        
        # Допустимые значения id
        allowed_ids = set(range(1, 8)) | {11}  # 1-7 и 11
        
        for attr_name, attr_data in attr_map.items():
            # Пропускаем исключенные названия
            if attr_name in excluded_names:
                continue
            
            if not isinstance(attr_data, dict):
                continue
            
            # Извлекаем id из структуры meta.typeMeta.id
            meta = attr_data.get("meta")
            if not isinstance(meta, dict):
                continue
            
            type_meta = meta.get("typeMeta")
            if not isinstance(type_meta, dict):
                continue
            
            attr_id = type_meta.get("id")
            if attr_id not in allowed_ids:
                continue
            
            # Извлекаем value
            value = attr_data.get("value")
            
            # Если id=4, то value - словарь, используем поле name
            if attr_id == 4:
                if isinstance(value, dict):
                    value = value.get("name")
                else:
                    continue  # Пропускаем, если value не словарь для id=4
            
            # Добавляем метаданные
            if value is not None:
                filtered_metadata[attr_name] = value
        
        return filtered_metadata
    
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
            
            # Добавляем метаданные из attrMap с фильтрацией
            if "attrMap" in irv_info and isinstance(irv_info["attrMap"], dict):
                filtered_attrs = self._filter_attr_map_metadata(irv_info["attrMap"])
                if filtered_attrs:
                    irv_metadata["irv_attrs"] = filtered_attrs
            
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
        supported_extensions = {".docx", ".txt", ".md"}
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

    def _get_cached_config(self):
        """Получение конфигурации с кэшированием."""
        if RAGService._config_cache is None:
            from utils.config import get_config
            RAGService._config_cache = get_config()
        return RAGService._config_cache
    
    def _get_cached_embedding(self, embed_api_key: Optional[str] = None, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None, embed_batch_size: Optional[int] = None):
        """
        Получение объекта эмбеддингов с кэшированием.
        
        Args:
            embed_api_key: API ключ для эмбеддингов (если None, берется из переменной окружения)
            embed_url: URL API эмбеддингов (если None, берется из конфигурации)
            embed_model_name: Модель эмбеддингов (если None, берется из конфигурации)
            embed_batch_size: Размер батча для эмбеддингов (если None, берется из конфигурации)
        """
        from rag.giga_embeddings import GigaEmbedding
        
        config = self._get_cached_config()
        embeddings_config = config.get("embeddings", {}).get("giga", {})
        
        # Определяем параметры: приоритет у параметров из запроса, затем конфигурация, затем переменные окружения
        # Проверяем, что ключ не пустой (если передан)
        if embed_api_key is not None and not embed_api_key.strip():
            raise ServiceError(
                error="Неверный API ключ для эмбеддингов",
                detail="API ключ указан в запросе, но является пустой строкой",
                code="empty_embed_api_key",
            )
        
        final_api_key = embed_api_key if embed_api_key and embed_api_key.strip() else os.getenv("GIGACHAT_AUTH_KEY")
        
        # Логируем источник ключа для отладки
        if embed_api_key and embed_api_key.strip():
            logger.debug("Используется API ключ из запроса (embed_api_key)")
        elif os.getenv("GIGACHAT_AUTH_KEY"):
            logger.debug("Используется API ключ из переменной окружения GIGACHAT_AUTH_KEY")
        else:
            logger.error("API ключ не найден ни в запросе, ни в переменных окружения")
        
        if not final_api_key:
            raise ServiceError(
                error="Не настроен API ключ для эмбеддингов",
                detail="API ключ не указан в запросе (embed_api_key) и переменная окружения GIGACHAT_AUTH_KEY не установлена",
                code="missing_embed_api_key",
            )
        
        final_api_url = embed_url or embeddings_config.get("api_url", "https://gigachat.devices.sberbank.ru/api/v1")
        final_model = embed_model_name or embeddings_config.get("model", "Embeddings")
        final_scope = embeddings_config.get("scope", "GIGACHAT_API_PERS")
        batch_size = embed_batch_size if embed_batch_size is not None else embeddings_config.get("batch_size", 10)
        max_retries = embeddings_config.get("max_retries", 3)
        timeout = embeddings_config.get("timeout", 60)
        
        # Логируем использование batch_size из запроса
        if embed_batch_size is not None:
            logger.debug(f"Используется batch_size из запроса: {embed_batch_size}")
        
        # Создаем ключ кэша на основе параметров
        cache_key = f"{final_api_url}:{final_model}:{final_scope}:{batch_size}:{max_retries}:{timeout}"
        # Не включаем api_key в ключ кэша для безопасности, но используем его при создании объекта
        
        if cache_key not in RAGService._embedding_cache:
            embedding = GigaEmbedding(
                credentials=final_api_key,
                scope=final_scope,
                api_url=final_api_url,
                model=final_model,
                batch_size=batch_size,
                max_retries=max_retries,
                timeout=timeout
            )
            RAGService._embedding_cache[cache_key] = embedding
            logger.debug(f"Создан новый объект GigaEmbedding для {final_api_url}/{final_model} (кэширован)")
        
        return RAGService._embedding_cache[cache_key]
    
    def _get_cached_vector_store(self, vdb_url: str, collection_name: str, vector_size: int, timeout: int, api_key: str = None):
        """Получение объекта векторного хранилища с кэшированием по ключу vdb_url."""
        # Нормализация URL для использования в качестве ключа кэша
        normalized_url = vdb_url.strip().rstrip("/")
        if not normalized_url.startswith("http"):
            normalized_url = f"http://{normalized_url}"
        
        cache_key = f"{normalized_url}:{collection_name}:{vector_size}"
        
        if cache_key not in RAGService._vector_store_cache:
            from rag.vector_store import QdrantVectorStoreManager
            
            vector_store_manager = QdrantVectorStoreManager(
                url=normalized_url,
                api_key=api_key,
                collection_name=collection_name,
                vector_size=vector_size,
                timeout=timeout
            )
            
            # Убеждаемся, что коллекция существует (только при первом создании)
            vector_store_manager.ensure_collection_exists()
            
            RAGService._vector_store_cache[cache_key] = vector_store_manager
            logger.debug(f"Создан новый объект QdrantVectorStoreManager для {normalized_url} (кэширован)")
        
        return RAGService._vector_store_cache[cache_key]
    
    def _initialize_rag_components(self, request: RAGRequest, chunker_output_dir: str):
        """Инициализация компонентов RAG (chunker, embedding, vector_store) с кэшированием."""
        from rag.chunker_integration import ChunkerIntegration
        
        # Загрузка конфигурации (кэшируется)
        config = self._get_cached_config()
        
        # Инициализация chunker (не кэшируется, так как зависит от chunker_output_dir)
        chunker_config = config.get("chunker", {})
        chunker = ChunkerIntegration(
            chunker_config_path=chunker_config.get("config_path", "smartchanker_config.json"),
            output_dir=chunker_output_dir
        )
        
        # Инициализация эмбеддингов (кэшируется с учетом параметров из запроса)
        embedding = self._get_cached_embedding(
            embed_api_key=getattr(request, 'embed_api_key', None),
            embed_url=getattr(request, 'embed_url', None),
            embed_model_name=getattr(request, 'embed_model_name', None),
            embed_batch_size=getattr(request, 'embed_batch_size', None)
        )
        
        # Инициализация векторного хранилища (кэшируется по vdb_url)
        qdrant_config = config.get("qdrant", {})
        vdb_url = request.vdb_url.strip().rstrip("/")
        if not vdb_url.startswith("http"):
            vdb_url = f"http://{vdb_url}"
        
        vector_store_manager = self._get_cached_vector_store(
            vdb_url=vdb_url,
            collection_name=qdrant_config.get("collection_name", "smart_rag_documents"),
            vector_size=qdrant_config.get("vector_size", 1024),
            timeout=qdrant_config.get("timeout", 30),
            api_key=qdrant_config.get("api_key")
        )
        
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
        temp_path: Path,
        max_chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Обработка одного файла.
        
        Args:
            max_chunk_size: Максимальный размер чанка в символах (если указан, переопределяет значение из конфига)
        """
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
                    "toc_chunks_count": 0,
                    "table_chunks_count": 0
                }
            
            # Сохранение файла во временную директорию
            temp_file_path = temp_path / file_name
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            
            # Обработка документа через chunker
            chunker_result = chunker.process_document(
                str(temp_file_path),
                document_id=f"{irv_id}_{irvf_id}",
                max_chunk_size=max_chunk_size
            )
            
            # Подготовка метаданных для сохранения
            doc_metadata = {
                "irv_id": irv_id,
                "irvf_id": irvf_id,
                "file_name": file_name,
                **irv_metadata
            }
            
            # Создание узлов из чанков
            nodes, toc_nodes, table_nodes = self._create_nodes_from_chunks(
                chunker_result,
                doc_metadata,
                irv_id,
                irvf_id
            )
            
            # Индексация узлов
            if nodes or toc_nodes or table_nodes:
#                all_nodes = nodes + toc_nodes + table_nodes
                all_nodes = nodes
                
                # Обрабатываем чанки порциями, чтобы не перегружать API
                # Размер порции равен batch_size эмбеддера (по умолчанию 10)
                # Это позволяет избежать проблем с ограничениями API и улучшает производительность
                batch_size = getattr(embedding, 'batch_size', 10)
                all_embeddings = []
                all_points = []
                
                # Обрабатываем узлы порциями
                for i in range(0, len(all_nodes), batch_size):
                    batch_nodes = all_nodes[i:i + batch_size]
                    batch_texts = [node.text for node in batch_nodes]
                    
                    try:
                        batch_embeddings = embedding._get_text_embeddings(batch_texts)
                    except (RuntimeError, ValueError) as e:
                        # Ошибка получения токена доступа или неверный формат ответа GigaChat
                        error_msg = str(e)
                        logger.error(f"Ошибка при получении эмбеддингов для файла {file_name} (порция {i//batch_size + 1}): {error_msg}")
                        raise ServiceError(
                            error="Ошибка получения эмбеддингов",
                            detail=f"Не удалось получить эмбеддинги для файла {file_name}: {error_msg}",
                            code="embedding_error"
                        )
                    
                    if len(batch_embeddings) != len(batch_nodes):
                        error_msg = f"Несоответствие количества эмбеддингов для порции {i//batch_size + 1} файла {file_name}: получено {len(batch_embeddings)}, ожидалось {len(batch_nodes)}"
                        logger.error(error_msg)
                        raise ServiceError(
                            error="Ошибка получения эмбеддингов",
                            detail=error_msg,
                            code="embedding_error"
                        )
                    
                    # Формируем точки для текущей порции
                    for node, emb in zip(batch_nodes, batch_embeddings):
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=emb,
                            payload={
                                "text": node.text,
                                **node.metadata
                            }
                        )
                        all_points.append(point)
                    
                    all_embeddings.extend(batch_embeddings)
                
                # Сохранение всех точек в Qdrant одной операцией
                if all_points:
                    vector_store_manager.client.upsert(
                        collection_name=vector_store_manager.collection_name,
                        points=all_points
                    )
                    
                    return {
                        "file_name": file_name,
                        "irvf_id": irvf_id,
                        "chunks_count": len(nodes),
                        "toc_chunks_count": len(toc_nodes),
                        "table_chunks_count": len(table_nodes),
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
                        "toc_chunks_count": 0,
                        "table_chunks_count": 0
                    }
            else:
                logger.warning(f"Не найдено чанков для файла {file_name}")
                return {
                    "file_name": file_name,
                    "irvf_id": irvf_id,
                    "chunks_count": 0,
                    "toc_chunks_count": 0,
                    "table_chunks_count": 0,
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
                "toc_chunks_count": 0,
                "table_chunks_count": 0
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
        table_nodes = []
        
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
        
        # Обработка чанков таблиц
        for idx, table_chunk_data in enumerate(chunker_result.get("table_chunks", [])):
            text = table_chunk_data.get("text", "")
            if not text or not text.strip():
                continue
            
            chunk_metadata = table_chunk_data.get("metadata", {})
            node_metadata = {
                **doc_metadata,
                **chunk_metadata,
                "chunk_index": idx,
                "chunk_type": "table"
            }
            
            node = TextNode(
                text=text.strip(),
                metadata=node_metadata,
                id_=f"{irv_id}_{irvf_id}_table_{idx}"
            )
            table_nodes.append(node)
        
        return nodes, toc_nodes, table_nodes

    def get_collections(self, request: CollectionListRequest) -> Dict[str, Any]:
        """
        Получение списка коллекций в векторной БД.
        
        Args:
            request: Запрос с параметрами vdb_url
            
        Returns:
            Словарь со списком коллекций
        """
        # Инициализация векторного хранилища (выносим за try для использования в except)
        vdb_url = request.vdb_url.strip().rstrip("/")
        if not vdb_url.startswith("http"):
            vdb_url = f"http://{vdb_url}"
        
        try:
            from rag.vector_store import QdrantVectorStoreManager
            from utils.config import get_config
            
            # Загрузка конфигурации
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            
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
            
        except ServiceError:
            # Пробрасываем ServiceError как есть
            raise
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            
            # Проверяем тип ошибки подключения
            if "timeout" in error_message.lower() or "Timeout" in error_type:
                logger.error(f"Таймаут подключения к Qdrant на {vdb_url}: {e}")
                raise ServiceError(
                    error="Qdrant недоступен",
                    detail=f"Таймаут подключения к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен и доступен.",
                    code="qdrant_timeout"
                )
            elif "connection" in error_message.lower() or "Connection" in error_type or "connect" in error_message.lower():
                logger.error(f"Ошибка подключения к Qdrant на {vdb_url}: {e}")
                raise ServiceError(
                    error="Qdrant недоступен",
                    detail=f"Не удалось подключиться к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен.",
                    code="qdrant_connection_error"
                )
            else:
                logger.exception(f"Ошибка при получении списка коллекций: {e}")
                raise ServiceError(
                    error="Ошибка при работе с Qdrant",
                    detail=f"Ошибка при получении списка коллекций: {error_message}",
                    code="qdrant_error"
                )

    def delete_collection(self, request: CollectionDeleteRequest) -> Dict[str, Any]:
        """
        Удаление коллекции из векторной БД.
        
        Args:
            request: Запрос с параметрами vdb_url и collection_name
            
        Returns:
            Словарь с результатом удаления коллекции
        """
        # Инициализация векторного хранилища (выносим за try для использования в except)
        vdb_url = request.vdb_url.strip().rstrip("/")
        if not vdb_url.startswith("http"):
            vdb_url = f"http://{vdb_url}"
        
        try:
            from rag.vector_store import QdrantVectorStoreManager
            from utils.config import get_config
            
            # Загрузка конфигурации
            config = get_config()
            qdrant_config = config.get("qdrant", {})
            
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
            
        except ServiceError:
            # Пробрасываем ServiceError как есть
            raise
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            
            # Проверяем тип ошибки подключения
            if "timeout" in error_message.lower() or "Timeout" in error_type:
                logger.error(f"Таймаут подключения к Qdrant на {vdb_url}: {e}")
                raise ServiceError(
                    error="Qdrant недоступен",
                    detail=f"Таймаут подключения к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен и доступен.",
                    code="qdrant_timeout"
                )
            elif "connection" in error_message.lower() or "Connection" in error_type or "connect" in error_message.lower():
                logger.error(f"Ошибка подключения к Qdrant на {vdb_url}: {e}")
                raise ServiceError(
                    error="Qdrant недоступен",
                    detail=f"Не удалось подключиться к Qdrant серверу на {vdb_url}. Убедитесь, что сервер запущен.",
                    code="qdrant_connection_error"
                )
            else:
                logger.exception(f"Ошибка при удалении коллекции: {e}")
                raise ServiceError(
                    error="Ошибка при работе с Qdrant",
                    detail=f"Ошибка при удалении коллекции: {error_message}",
                    code="qdrant_error"
                )
