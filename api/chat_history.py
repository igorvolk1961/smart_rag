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


def load_chat_history(siu_client: Any, chat_history_irv_id: Optional[str]) -> Optional[List[dict]]:
    """
    Загружает историю чата из файла chat_history.json, приложенного к версии ИО с id chat_history_irv_id.

    Возвращает список сообщений для контекста LLM [{role, content}] или None, если идентификатор не задан
    или файл не найден/не удалось распарсить.
    """
    if not chat_history_irv_id or not str(chat_history_irv_id).strip():
        return None
    try:
        raw_files = siu_client.get_irv_files(chat_history_irv_id)
        files_list = _files_list(raw_files)
        file_obj = _find_file_by_name(files_list, CHAT_HISTORY_FILENAME)
        if not file_obj:
            logger.warning("Файл {} не найден у ИО версии {}", CHAT_HISTORY_FILENAME, chat_history_irv_id)
            return None
        content = siu_client.get_irv_file_content(file_obj)
        if content is None:
            return None
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
                return None
        elif isinstance(content, dict):
            parsed = content
        else:
            return None
        messages = _normalize_messages_from_json(parsed)
        if not messages:
            return None
        return messages
    except ServiceError:
        raise
    except Exception as e:
        logger.warning("Ошибка загрузки истории чата: {}", e)
        return None


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
) -> Optional[dict]:
    """
    Сохраняет историю чата в СИУ: либо создаёт новый ИО с файлом chat_history.json в папке
    «Диалоги с ИИ-помощником», либо дополняет существующий файл и создаёт новую версию ИО.

    full_messages — список сообщений для записи в chat_history.json (включая системный промпт, запрос и ответ).
    
    Returns:
        Словарь с результатом создания/обновления ИО или None при ошибке.
    """
    if chat_history_irv_id and str(chat_history_irv_id).strip():
        return _save_chat_history_update(
            siu_client,
            chat_history_irv_id=chat_history_irv_id,
            chat_title=chat_title,
            chat_summary=chat_summary or "",
            full_messages=full_messages,
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
        raw_files = siu_client.get_irv_files(new_irv_id)
        files_list = _files_list(raw_files)
        file_obj = _find_file_by_name(files_list, CHAT_HISTORY_FILENAME)
        if not file_obj:
            logger.warning("save_chat_history: файл {} не найден у созданного ИР {}", CHAT_HISTORY_FILENAME, new_irv_id)
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
) -> Optional[dict]:
    """Читает текущий chat_history.json, дополняет запросом и ответом, создаёт новую версию ИО и записывает файл.
    
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
        existing_name = current_irv.get("name") or (current_irv.get("description", "")) or "Диалог"
        # Извлекаем базовое имя (без timestamp) и добавляем новый timestamp
        if "#" in existing_name:
            base_name = existing_name.rsplit("#", 1)[0]
        else:
            base_name = existing_name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        updated_name = f"{base_name}#{timestamp}"
        if not io_id or not parent_id or not nau_id:
            logger.warning(
                "save_chat_history (update): не удалось извлечь io_id/parent_id/nau_id из ИО версии {}",
                chat_history_irv_id,
            )
            return None
        # full_messages уже содержит всю историю (загруженную через load_chat_history + текущий запрос + ответ)
        # Не нужно повторно читать файл - используем full_messages напрямую
        body_str = json.dumps({"messages": full_messages}, ensure_ascii=False, indent=2)
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
        raw_files = siu_client.get_irv_files(new_irv_id)
        files_list = _files_list(raw_files)
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
