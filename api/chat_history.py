"""
Загрузка и сохранение истории чата в СИУ (файл chat_history.json при версии ИО).
"""

import json
from datetime import datetime
from typing import Any, List, Optional

from loguru import logger

from api.exceptions import ServiceError

CHAT_HISTORY_FILENAME = "chat_history.json"
DIALOGS_FOLDER_NAME = "Диалоги с ИИ-помощником"


def _extract_parent_id(irv_data: dict) -> Optional[str]:
    """Из объекта версии ИО извлекает идентификатор родительской папки (parent_id / parentId)."""
    ir = irv_data.get("ir")
    if isinstance(ir, dict):
        pid = ir.get("parentId") or ir.get("parent_id")
        if pid:
            return str(pid)
    return None


def _extract_nau_id(irv_data: dict) -> Optional[str]:
    """Из объекта версии ИО извлекает nauId."""
    ir = irv_data.get("ir")
    if isinstance(ir, dict):
        nid = ir.get("nauId") or ir.get("nau_id")
        if nid:
            return str(nid)
    return None


def _extract_io_id(irv_data: dict) -> Optional[str]:
    """Из объекта версии ИО извлекает id информационного объекта (ioId)."""
    ir = irv_data.get("ir")
    if isinstance(ir, dict):
        return str(ir["id"]) if ir.get("id") else None
    return None


def _files_list(raw: Any) -> list:
    """Нормализует ответ get_irv_files в список элементов с полями name, irvfId."""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("contents", [])
        if not isinstance(items, list):
            items = [raw]
    else:
        items = []
    result = []
    for item in items:
        if isinstance(item, dict) and (item.get("name") or item.get("irvfId")):
            result.append(item)
    return result


def _find_file_by_name(files_list: list, name: str) -> Optional[dict]:
    """В списке файлов ищет элемент с полем name == name."""
    for f in files_list:
        n = f.get("name") or (f.get("fileName"))
        if n == name:
            return f
    return None


def _normalize_messages_from_json(parsed: Any) -> List[dict]:
    """Приводит распарсенный JSON к списку сообщений [{role, content}]."""
    if isinstance(parsed, list):
        out = []
        for m in parsed:
            if isinstance(m, dict) and m.get("role") and m.get("content") is not None:
                out.append({"role": m["role"], "content": m.get("content", "")})
        return out
    if isinstance(parsed, dict):
        messages = parsed.get("messages") or parsed.get("messages_list")
        if isinstance(messages, list):
            return _normalize_messages_from_json(messages)
    return []


def load_chat_history(siu_client: Any, chat_history_irv_id: Optional[str]) -> tuple[Optional[List[dict]], bool]:
    """
    Загружает историю чата из файла chat_history.json, приложенного к версии ИО с id chat_history_irv_id.

    Возвращает кортеж (messages, irv_exists):
    - messages: список сообщений для контекста LLM [{role, content}] или None (если файл не найден или пуст)
    - irv_exists: True если ИО существует (признак того, что нужно создавать новую версию)
    
    Если идентификатор не задан, возвращает (None, False).
    """
    if not chat_history_irv_id or not str(chat_history_irv_id).strip():
        return (None, False)
    try:
        # Сначала проверяем существование ИО
        irv = siu_client.get_irv(chat_history_irv_id)
        irv_exists = isinstance(irv, dict)
        if not irv_exists:
            logger.warning("ИО версии {} не существует", chat_history_irv_id)
            return (None, False)
        
        # Проверяем наличие файла
        raw_files = siu_client.get_irv_files(chat_history_irv_id)
        files_list = _files_list(raw_files)
        file_obj = _find_file_by_name(files_list, CHAT_HISTORY_FILENAME)
        if not file_obj:
            logger.warning("Файл {} не найден у ИО версии {}", CHAT_HISTORY_FILENAME, chat_history_irv_id)
            return (None, True)
        
        # Загружаем содержимое файла
        content = siu_client.get_irv_file_content(file_obj)
        if content is None:
            return (None, True)
        if isinstance(content, (bytes, bytearray)):
            content = content.decode("utf-8", errors="replace")
        if isinstance(content, dict):
            content = content.get("content") or content
        if isinstance(content, (bytes, bytearray)):
            content = content.decode("utf-8", errors="replace")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Не удалось распарсить {} как JSON", CHAT_HISTORY_FILENAME)
                return (None, True)
        elif isinstance(content, dict):
            parsed = content
        else:
            return (None, True)
        messages = _normalize_messages_from_json(parsed)
        if not messages:
            return (None, True)
        return (messages, True)
    except ServiceError:
        raise
    except Exception as e:
        logger.warning("Ошибка загрузки истории чата: {}", e)
        return (None, False)


def _extract_irv_id_from_response(raw: Any) -> Optional[str]:
    """Из ответа create_ir извлекает id созданной версии ИО."""
    if not raw:
        return None
    if isinstance(raw, dict) and raw.get("id"):
        return str(raw["id"])
    return None


def _extract_folder_id_from_response(raw: Any) -> Optional[str]:
    """Из ответа create_folder / get_folder_children извлекает id папки."""
    if not raw:
        return None
    if isinstance(raw, dict) and raw.get("id"):
        return str(raw["id"])
    return None


def _children_list(raw: Any) -> list:
    """Нормализует ответ get_folder_children в список элементов с полями id, name."""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("contents", [])
        if not isinstance(items, list):
            items = [raw]
    else:
        items = []
    return items


def _find_child_folder_by_name(children: list, name: str) -> Optional[dict]:
    """Ищет среди дочерних элементов папку с полем name == name."""
    for item in children:
        if isinstance(item, dict) and (item.get("name") or item.get("fileName")) == name:
            return item
    return None


def save_chat_history(
    siu_client: Any,
    *,
    chat_history_irv_id: Optional[str],
    irv_id: Optional[str],
    chat_title: Optional[str],
    chat_summary: Optional[str],
    full_messages: List[dict],
    irv_exists: bool = False,
    has_messages: bool = False,
) -> Optional[dict]:
    """
    Сохраняет историю чата в СИУ: либо создаёт новый ИО с файлом chat_history.json в папке
    «Диалоги с ИИ-помощником», либо дополняет существующий файл и создаёт новую версию ИО.

    full_messages — список сообщений для записи в chat_history.json (включая системный промпт, запрос и ответ).
    irv_exists — существует ли ИО (признак того, что нужно создавать новую версию).
    has_messages — были ли загружены сообщения из истории (признак того, как формировать имя ИО).
    
    Returns:
        Словарь с результатом создания/обновления ИО или None при ошибке.
    """
    if chat_history_irv_id and str(chat_history_irv_id).strip() and irv_exists:
        return _save_chat_history_update(
            siu_client,
            chat_history_irv_id=chat_history_irv_id,
            chat_title=chat_title,
            chat_summary=chat_summary or "",
            full_messages=full_messages,
            has_messages=has_messages,
        )
    else:
        return _save_chat_history_new(
            siu_client,
            irv_id=irv_id,
            chat_title=chat_title,
            chat_summary=chat_summary,
            full_messages=full_messages,
        )


def _save_chat_history_new(
    siu_client: Any,
    *,
    irv_id: Optional[str],
    chat_title: Optional[str],
    chat_summary: Optional[str],
    full_messages: List[dict],
) -> Optional[dict]:
    """Создаёт новый ИО диалога в папке «Диалоги с ИИ-помощником» и прикладывает chat_history.json.
    
    Returns:
        Словарь с результатом создания ИО или None при ошибке.
    """
    if not irv_id or not str(irv_id).strip():
        logger.warning("save_chat_history: irv_id не задан, сохранение пропущено")
        return None
    try:
        current_irv = siu_client.get_irv(irv_id)
        if not isinstance(current_irv, dict):
            logger.warning("save_chat_history: get_irv вернул не dict")
            return None
        parent_id = _extract_parent_id(current_irv)
        nau_id = _extract_nau_id(current_irv)
        if not parent_id or not nau_id:
            logger.warning("save_chat_history: не удалось извлечь parent_id или nau_id из ИО {}", irv_id)
            return None
        raw_children = siu_client.get_folder_children(parent_id)
        children = _children_list(raw_children)
        dialogs_folder = _find_child_folder_by_name(children, DIALOGS_FOLDER_NAME)
        if dialogs_folder:
            dialogs_folder_id = _extract_folder_id_from_response(dialogs_folder) or dialogs_folder.get("id")
        else:
            folder_description = (
                "Папка содержит ИО, к которым прикреплены файлы с сохраненными диалогами с ИИ-помощником. "
                "Чтобы продолжить диалог, укажите ссылку на него в поле \"Сохраненный контекст диалога\""
            )
            created = siu_client.create_folder(DIALOGS_FOLDER_NAME, parent_id, description=folder_description)
            dialogs_folder_id = _extract_folder_id_from_response(created)
        if not dialogs_folder_id:
            logger.warning("save_chat_history: не удалось получить dialogs_folder_id")
            return None
        base_title = (chat_title or "").strip() or (full_messages[0].get("content", "")[:80] if full_messages else "Диалог")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        title = f"{base_title}#{timestamp}"
        summary = (chat_summary or "").strip() or ""
        body_str = json.dumps({"messages": full_messages}, ensure_ascii=False, indent=2)
        create_result = siu_client.create_ir(
            irv_name=title,
            parent_folder_id=dialogs_folder_id,
            nau_id=nau_id,
            description=summary,
            file_name=CHAT_HISTORY_FILENAME,
        )
        new_irv_id = _extract_irv_id_from_response(create_result)
        if not new_irv_id:
            logger.warning("save_chat_history: не удалось извлечь id созданного ИР из ответа create_ir")
            return None
        # Используем get_irv с with_files=True для получения информации о файлах
        new_irv = siu_client.get_irv(new_irv_id, with_files=True)
        if not isinstance(new_irv, dict):
            logger.warning("save_chat_history (new): get_irv вернул не dict для созданного ИР")
            return None
        # Извлекаем файлы из ответа get_irv
        files_attr = new_irv.get("attrMap", {})
        if not isinstance(files_attr, dict):
            logger.warning("save_chat_history (new): attrMap не найден или не является dict")
            return None
        files_value = files_attr.get("Файлы", {})
        if not isinstance(files_value, dict):
            logger.warning("save_chat_history (new): поле 'Файлы' не найдено или не является dict")
            return None
        files_data = files_value.get("value", [])
        if not isinstance(files_data, list):
            logger.warning("save_chat_history (new): value поля 'Файлы' не является list")
            return None
        files_list = _files_list(files_data)
        file_obj = _find_file_by_name(files_list, CHAT_HISTORY_FILENAME)
        if not file_obj:
            logger.warning("save_chat_history (new): файл {} не найден у созданного ИР {}", CHAT_HISTORY_FILENAME, new_irv_id)
            return None
        siu_client.post_irv_file_content(file_obj, body_str)
        # Возвращаем результат создания ИО
        return create_result if isinstance(create_result, dict) else None
    except ServiceError:
        raise
    except Exception as e:
        logger.exception("Ошибка сохранения истории чата (новый диалог): {}", e)
        raise


def _save_chat_history_update(
    siu_client: Any,
    *,
    chat_history_irv_id: str,
    chat_title: Optional[str],
    chat_summary: str,
    full_messages: List[dict],
    has_messages: bool = False,
) -> Optional[dict]:
    """Создаёт новую версию ИО и записывает файл chat_history.json.
    
    has_messages — были ли загружены сообщения из истории (если False, имя ИО формируется как для нового).
    
    Returns:
        Словарь с результатом создания новой версии ИО или None при ошибке.
    """
    try:
        current_irv = siu_client.get_irv(chat_history_irv_id)
        if not isinstance(current_irv, dict):
            logger.warning("save_chat_history (update): get_irv вернул не dict")
            return None
        io_id = _extract_io_id(current_irv)
        parent_id = _extract_parent_id(current_irv)
        nau_id = _extract_nau_id(current_irv)
        if not io_id or not parent_id or not nau_id:
            logger.warning(
                "save_chat_history (update): не удалось извлечь io_id/parent_id/nau_id из ИО версии {}",
                chat_history_irv_id,
            )
            return None
        # full_messages уже содержит всю историю (загруженную через load_chat_history + текущий запрос + ответ)
        body_str = json.dumps({"messages": full_messages}, ensure_ascii=False, indent=2)
        if has_messages:
            # Файл существует - создаём новую версию ИО с обновлённым файлом
            # Название формируется из существующего имени с новым timestamp
            existing_name = current_irv.get("name") or (current_irv.get("description", "")) or "Диалог"
            if "#" in existing_name:
                base_name = existing_name.rsplit("#", 1)[0]
            else:
                base_name = existing_name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            updated_name = f"{base_name}#{timestamp}"
        else:
            # Файл отсутствует - создаём новую версию ИО с файлом
            # Название формируется как при создании нового ИО (chat_title#timestamp)
            base_title = (chat_title or "").strip() or (full_messages[0].get("content", "")[:80] if full_messages else "Диалог")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            updated_name = f"{base_title}#{timestamp}"
        
        create_result = siu_client.create_ir(
            irv_name=updated_name,
            parent_folder_id=parent_id,
            nau_id=nau_id,
            description=chat_summary,
            file_name=CHAT_HISTORY_FILENAME,
            io_id=io_id,
        )
        new_irv_id = _extract_irv_id_from_response(create_result)
        if not new_irv_id:
            logger.warning("save_chat_history (update): не удалось извлечь id новой версии ИР")
            return None
        # Используем get_irv с with_files=True для получения информации о файлах
        new_irv = siu_client.get_irv(new_irv_id, with_files=True)
        if not isinstance(new_irv, dict):
            logger.warning("save_chat_history (update): get_irv вернул не dict для новой версии ИР")
            return None
        # Извлекаем файлы из ответа get_irv
        files_attr = new_irv.get("attrMap", {})
        if not isinstance(files_attr, dict):
            logger.warning("save_chat_history (update): attrMap не найден или не является dict")
            return None
        files_value = files_attr.get("Файлы", {})
        if not isinstance(files_value, dict):
            logger.warning("save_chat_history (update): поле 'Файлы' не найдено или не является dict")
            return None
        files_data = files_value.get("value", [])
        if not isinstance(files_data, list):
            logger.warning("save_chat_history (update): value поля 'Файлы' не является list")
            return None
        files_list = _files_list(files_data)
        file_obj = _find_file_by_name(files_list, CHAT_HISTORY_FILENAME)
        if not file_obj:
            logger.warning("save_chat_history (update): файл {} не найден у новой версии ИР", CHAT_HISTORY_FILENAME)
            return None
        siu_client.post_irv_file_content(file_obj, body_str)
        # Возвращаем результат создания новой версии ИО
        return create_result if isinstance(create_result, dict) else None
    except ServiceError:
        raise
    except Exception as e:
        logger.exception("Ошибка сохранения истории чата (обновление): {}", e)
        raise


def save_result_file(
    siu_client: Any,
    result_irv_id: str,
    content: str,
    file_name: str,
) -> Optional[dict]:
    """
    Сохранение файла с текстом ответа в указанный ИО.
    
    Args:
        siu_client: Клиент для работы с СИУ
        result_irv_id: Идентификатор версии информационного объекта для сохранения файла ответа
        content: Текст ответа для сохранения
        file_name: Имя файла (с расширением .txt)
    
    Returns:
        Результат создания новой версии ИО или None при ошибке
    """
    if not result_irv_id or not str(result_irv_id).strip():
        logger.warning("save_result_file: result_irv_id не задан")
        return None
    
    if not content or not str(content).strip():
        logger.warning("save_result_file: содержимое файла пусто")
        return None
    
    if not file_name or not str(file_name).strip():
        logger.warning("save_result_file: имя файла не задано")
        return None
    
    # Убеждаемся, что имя файла имеет расширение .txt
    file_name = str(file_name).strip()
    if not file_name.endswith('.txt'):
        file_name = f"{file_name}.txt"
    
    try:
        # Получаем информацию об ИО
        irv_info = siu_client.get_irv(result_irv_id, with_files=True)
        if not isinstance(irv_info, dict):
            logger.warning("save_result_file: не удалось получить информацию об ИО {}", result_irv_id)
            return None
        
        # Извлекаем io_id и другие необходимые данные
        io_id = _extract_io_id(irv_info)
        if not io_id:
            logger.warning("save_result_file: не удалось извлечь io_id из ИО {}", result_irv_id)
            return None
        
        # Извлекаем parent_id и nau_id для создания новой версии
        parent_id = _extract_parent_id(irv_info)
        nau_id = _extract_nau_id(irv_info)
        
        if not parent_id:
            logger.warning("save_result_file: не удалось извлечь parent_id из ИО {}", result_irv_id)
            return None
        
        if not nau_id:
            logger.warning("save_result_file: не удалось извлечь nau_id из ИО {}", result_irv_id)
            return None
        
        # Создаем новую версию ИО с файлом
        # Используем имя файла без расширения как название версии
        version_name = file_name.replace('.txt', '')
        create_result = siu_client.create_ir(
            irv_name=version_name,
            parent_folder_id=parent_id,
            nau_id=nau_id,
            description=f"Ответ сохранен в файле {file_name}",
            file_name=file_name,
            io_id=io_id,  # Создаем новую версию существующего ИО
        )
        
        new_irv_id = _extract_irv_id_from_response(create_result)
        if not new_irv_id:
            logger.warning("save_result_file: не удалось извлечь id созданной версии ИО")
            return None
        
        # Получаем информацию о файле из новой версии
        new_irv = siu_client.get_irv(new_irv_id, with_files=True)
        if not isinstance(new_irv, dict):
            logger.warning("save_result_file: get_irv вернул не dict для новой версии ИО {}", new_irv_id)
            return None
        
        # Извлекаем файлы из ответа get_irv
        files_attr = new_irv.get("attrMap", {})
        if not isinstance(files_attr, dict):
            logger.warning("save_result_file: attrMap не найден или не является dict")
            return None
        
        files_value = files_attr.get("Файлы", {})
        if not isinstance(files_value, dict):
            logger.warning("save_result_file: поле 'Файлы' не найдено или не является dict")
            return None
        
        files_data = files_value.get("value", [])
        if not isinstance(files_data, list):
            logger.warning("save_result_file: value поля 'Файлы' не является list")
            return None
        
        files_list = _files_list(files_data)
        file_obj = _find_file_by_name(files_list, file_name)
        if not file_obj:
            logger.warning("save_result_file: файл {} не найден у новой версии ИО {}", file_name, new_irv_id)
            return None
        
        # Сохраняем содержимое файла
        siu_client.post_irv_file_content(file_obj, content)
        logger.info("Файл ответа {} успешно сохранен в ИО {}", file_name, new_irv_id)
        
        return create_result if isinstance(create_result, dict) else None
        
    except ServiceError:
        raise
    except Exception as e:
        logger.exception("Ошибка сохранения файла ответа: {}", e)
        raise
