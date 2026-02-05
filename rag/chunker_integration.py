"""
Интеграция SmartChanker для обработки документов.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from smart_chanker.smart_chanker import SmartChanker
except ImportError:
    raise ImportError(
        "SmartChanker не установлен. Установите его командой: "
        "pip install git+https://github.com/igorvolk1961/smart_chanker.git"
    )

logger = logging.getLogger(__name__)


class ChunkerIntegration:
    """
    Класс для интеграции SmartChanker в RAG-систему.
    
    Обрабатывает документы через SmartChanker и подготавливает
    чанки с метаданными для индексации в векторное хранилище.
    """
    
    def __init__(self, chunker_config_path: str, output_dir: str = "data/chunks"):
        """
        Инициализация интеграции с SmartChanker.
        
        Args:
            chunker_config_path: Путь к конфигурационному файлу SmartChanker
            output_dir: Директория для сохранения результатов чанкинга
        """
        self.chunker_config_path = Path(chunker_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация SmartChanker
        if not self.chunker_config_path.exists():
            logger.warning(
                f"Конфигурационный файл SmartChanker не найден: {chunker_config_path}. "
                "Создайте файл config.json с настройками чанкера."
            )
        
        self.chunker = SmartChanker(str(self.chunker_config_path))
        logger.info(f"SmartChanker инициализирован с конфигом: {chunker_config_path}")
    
    def process_document(
        self, 
        document_path: str, 
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Обработка документа через SmartChanker.
        
        Args:
            document_path: Путь к документу для обработки
            document_id: Уникальный идентификатор документа (если None, генерируется из имени файла)
        
        Returns:
            Словарь с результатами обработки:
            - chunks: список чанков с метаданными
            - metadata: общие метаданные документа
            - toc_chunks: чанки оглавления (если есть)
        """
        doc_path = Path(document_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Документ не найден: {document_path}")
        
        if document_id is None:
            document_id = doc_path.stem
        
        logger.info(f"Обработка документа: {document_path} (ID: {document_id})")
        
        # Создание выходной директории для этого документа
        doc_output_dir = self.output_dir / document_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Обработка документа через SmartChanker
        try:
            result = self.chunker.run_end_to_end(
                str(doc_path),
                str(doc_output_dir)
            )
            
            logger.info(f"Документ обработан успешно. Результаты сохранены в: {doc_output_dir}")
            
            # Проверяем, что вернул SmartChanker
            if result is not None:
                logger.debug(f"SmartChanker вернул результат типа: {type(result)}")
                if isinstance(result, dict):
                    logger.debug(f"Ключи в результате: {result.keys()}")
            
            # Загрузка и парсинг результатов
            # Сначала пробуем использовать результат напрямую, если он есть
            if result is not None and isinstance(result, dict):
                chunks_data = self._load_chunks_from_dict(result, document_id)
            else:
                # Иначе загружаем из файлов
                chunks_data = self._load_chunks_from_result(doc_output_dir, document_id)
            
            return {
                "document_id": document_id,
                "document_path": str(doc_path),
                "chunks": chunks_data["chunks"],
                "metadata": chunks_data["metadata"],
                "toc_chunks": chunks_data.get("toc_chunks", []),
                "table_chunks": chunks_data.get("table_chunks", []),
                "output_dir": str(doc_output_dir)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документа {document_path}: {e}", exc_info=True)
            raise
    
    def _load_chunks_from_dict(
        self,
        result_dict: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Загрузка чанков из словаря результата SmartChanker.
        
        Args:
            result_dict: Словарь с результатами SmartChanker
            document_id: ID документа
        
        Returns:
            Словарь с чанками и метаданными
        """
        chunks = []
        metadata = {
            "document_id": document_id,
            "total_chunks": 0
        }
        toc_chunks = []
        
        # Извлечение чанков из результата
        if isinstance(result_dict, dict):
            # Пробуем разные возможные ключи
            raw_chunks = (
                result_dict.get("chunks") or
                result_dict.get("hierarchical_chunks") or
                result_dict.get("data") or
                []
            )
            
            # Обработка чанков
            for idx, chunk_data in enumerate(raw_chunks):
                chunk = self._process_chunk(chunk_data, idx, document_id)
                if chunk:
                    chunks.append(chunk)
            
            # Извлечение TOC чанков
            if "toc_chunks" in result_dict:
                for idx, toc_chunk_data in enumerate(result_dict["toc_chunks"]):
                    toc_chunk = self._process_chunk(toc_chunk_data, idx, document_id, is_toc=True)
                    if toc_chunk:
                        toc_chunks.append(toc_chunk)
            
            metadata["total_chunks"] = len(chunks)
            metadata["has_toc"] = len(toc_chunks) > 0
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "toc_chunks": toc_chunks,
            "table_chunks": result_dict.get("table_chunks", []) if isinstance(result_dict, dict) else []
        }
    
    def _load_chunks_from_result(
        self, 
        output_dir: Path, 
        document_id: str
    ) -> Dict[str, Any]:
        """
        Загрузка чанков из результатов SmartChanker.
        
        Args:
            output_dir: Директория с результатами обработки
            document_id: ID документа
        
        Returns:
            Словарь с чанками и метаданными
        """
        chunks = []
        metadata = {
            "document_id": document_id,
            "total_chunks": 0
        }
        toc_chunks = []
        table_chunks = []
        
        # Поиск JSON файла с результатами
        json_files = list(output_dir.glob("*.json"))
        
        # Если JSON файлов нет, пробуем найти текстовые файлы с чанками
        if not json_files:
            logger.warning(f"JSON файлы с результатами не найдены в {output_dir}")
            
            # Пробуем найти текстовые файлы (SmartChanker может создавать .txt файлы)
            txt_files = [f for f in output_dir.glob("*.txt") if not f.name.endswith("_toc.txt")]
            
            if txt_files:
                logger.info(f"Найдены текстовые файлы: {[f.name for f in txt_files]}")
                # Читаем текстовые файлы как чанки
                for idx, txt_file in enumerate(txt_files):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                chunk = {
                                    "text": text,
                                    "metadata": {
                                        "document_id": document_id,
                                        "chunk_index": idx,
                                        "source_file": txt_file.name
                                    }
                                }
                                chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Ошибка при чтении файла {txt_file}: {e}")
                
                metadata["total_chunks"] = len(chunks)
                return {
                    "chunks": chunks,
                    "metadata": metadata,
                    "toc_chunks": toc_chunks
                }
            else:
                logger.warning("Не найдены ни JSON, ни текстовые файлы с чанками")
                return {
                    "chunks": chunks,
                    "metadata": metadata,
                    "toc_chunks": toc_chunks
                }
        
        # Загрузка основного JSON файла (обычно первый найденный)
        main_json_file = json_files[0]
        
        try:
            with open(main_json_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Извлечение чанков из результата
            # Структура зависит от формата вывода SmartChanker
            if isinstance(result_data, dict):
                # Если результат - словарь с ключом chunks или подобным
                if "chunks" in result_data:
                    raw_chunks = result_data["chunks"]
                elif "hierarchical_chunks" in result_data:
                    raw_chunks = result_data["hierarchical_chunks"]
                else:
                    # Пытаемся найти чанки в других ключах
                    raw_chunks = result_data.get("data", [])
                
                # Обработка чанков
                for idx, chunk_data in enumerate(raw_chunks):
                    chunk = self._process_chunk(chunk_data, idx, document_id)
                    if chunk:
                        chunks.append(chunk)
                
                # Извлечение TOC чанков
                if "toc_chunks" in result_data:
                    for idx, toc_chunk_data in enumerate(result_data["toc_chunks"]):
                        toc_chunk = self._process_chunk(toc_chunk_data, idx, document_id, is_toc=True)
                        if toc_chunk:
                            toc_chunks.append(toc_chunk)
                
                # Извлечение table_chunks из результата (если есть)
                if "table_chunks" in result_data:
                    for idx, table_chunk_data in enumerate(result_data["table_chunks"]):
                        table_chunk = self._process_chunk(table_chunk_data, idx, document_id, is_table=True)
                        if table_chunk:
                            table_chunks.append(table_chunk)
                
                # Извлечение метаданных
                metadata.update({
                    "total_chunks": len(chunks),
                    "has_toc": len(toc_chunks) > 0,
                    "has_tables": len(table_chunks) > 0,
                    "source_file": str(main_json_file)
                })
                
            elif isinstance(result_data, list):
                # Если результат - список чанков
                for idx, chunk_data in enumerate(result_data):
                    chunk = self._process_chunk(chunk_data, idx, document_id)
                    if chunk:
                        chunks.append(chunk)
                
                metadata["total_chunks"] = len(chunks)
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке чанков из {main_json_file}: {e}", exc_info=True)
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "toc_chunks": toc_chunks,
            "table_chunks": table_chunks if 'table_chunks' in locals() else []
        }
    
    def _process_chunk(
        self, 
        chunk_data: Any, 
        index: int, 
        document_id: str,
        is_toc: bool = False,
        is_table: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Обработка одного чанка и извлечение метаданных.
        
        Args:
            chunk_data: Данные чанка из SmartChanker
            index: Индекс чанка
            document_id: ID документа
            is_toc: Флаг, указывающий что это чанк оглавления
        
        Returns:
            Словарь с обработанным чанком и метаданными
        """
        try:
            if isinstance(chunk_data, dict):
                # Извлечение текста
                text = chunk_data.get("text", chunk_data.get("content", ""))
                
                if not text or not text.strip():
                    return None
                
                # Извлечение метаданных
                metadata = {
                    "document_id": document_id,
                    "chunk_index": index,
                    "is_toc": is_toc,
                    "is_table": is_table,
                    "hierarchy_level": chunk_data.get("level", chunk_data.get("hierarchy_level")),
                    "section_number": chunk_data.get("section_number", chunk_data.get("number")),
                    "parent_section": chunk_data.get("parent_section"),
                    "sibling_index": chunk_data.get("sibling_index"),
                    "position": chunk_data.get("position"),
                    "chunk_type": chunk_data.get("type", "table" if is_table else ("toc" if is_toc else "text"))
                }
                
                # Удаление None значений
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                return {
                    "text": text.strip(),
                    "metadata": metadata
                }
            
            elif isinstance(chunk_data, str):
                # Если чанк - просто строка
                return {
                    "text": chunk_data.strip(),
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": index,
                        "is_toc": is_toc,
                        "is_table": is_table,
                        "chunk_type": "table" if is_table else ("toc" if is_toc else "text")
                    }
                }
            
            else:
                logger.warning(f"Неожиданный формат чанка: {type(chunk_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при обработке чанка {index}: {e}", exc_info=True)
            return None
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Обработка всех документов в папке.
        
        Args:
            folder_path: Путь к папке с документами
        
        Returns:
            Список результатов обработки для каждого документа
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")
        
        results = []
        
        # Поддерживаемые форматы
        supported_formats = [".docx", ".txt", ".pdf"]
        
        for doc_file in folder.iterdir():
            if doc_file.is_file() and doc_file.suffix.lower() in supported_formats:
                try:
                    result = self.process_document(str(doc_file))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {doc_file}: {e}", exc_info=True)
        
        logger.info(f"Обработано документов: {len(results)} из {len(list(folder.iterdir()))}")
        
        return results

