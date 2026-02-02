"""
Клиент REST API СИУ (единый базовый URL, аутентификация по JSESSIONID).
Содержит методы для запросов к сервисам СИУ; при необходимости добавляются новые GET/POST.
Соответствует функциям из siu_api.js (reportSiu* / doCallSiu).
"""

import hashlib
import json
from typing import Any, List, Optional, Union
from urllib.parse import quote

import httpx

from api.exceptions import ServiceError

# Базовый путь API СИУ (добавляется к _base_url)
_SIU_API_PATH = "/siu-star/services/api"


class SiuClient:
    """
    Клиент для запросов к API СИУ с общим базовым URL и cookie JSESSIONID.
    Все запросы идут на base_url + path; при недостатке данных или ошибке — ServiceError.
    """

    def __init__(
        self,
        base_url: str,
        jsessionid: Optional[str],
        *,
        timeout: float = 10.0,
    ):
        if not (base_url or "").strip():
            raise ServiceError(
                error="Недостаточно информации для выполнения запроса к СИУ",
                detail="В заголовках запроса отсутствует или пуст referer.",
                status_code=400,
                code="missing_referer",
            )
        if not jsessionid or (isinstance(jsessionid, str) and not jsessionid.strip()):
            raise ServiceError(
                error="Недостаточно информации для запроса к СИУ",
                detail="В cookie отсутствует JSESSIONID.",
                status_code=401,
                code="missing_jsessionid",
            )
        self._base_url = (base_url or "").rstrip("/")
        self._api_base = self._base_url + _SIU_API_PATH
        self._cookies = {"JSESSIONID": jsessionid.strip()}
        self._timeout = timeout

    def _get(
        self,
        path: str,
        *,
        error_label: str = "запрос",
        error_code: str = "siu_error",
    ) -> Any:
        """GET api_base + path, возвращает response.json() (dict или list). При ошибке — ServiceError."""
        url = self._api_base + path
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, cookies=self._cookies)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise ServiceError(
                error=f"Ошибка сервиса СИУ ({error_label})",
                detail=f"Сервис вернул {e.response.status_code}: {e.response.text[:200] if e.response.text else ''}",
                status_code=e.response.status_code,
                code=error_code,
            ) from e
        except httpx.HTTPError as e:
            raise ServiceError(
                error=f"Ошибка соединения с сервисом СИУ ({error_label})",
                detail=str(e),
                status_code=503,
                code="siu_connection_error",
            ) from e
        except json.JSONDecodeError as e:
            raise ServiceError(
                error=f"Ошибка ответа сервиса СИУ ({error_label})",
                detail=f"Некорректный JSON в ответе: {e}",
                status_code=502,
                code="siu_invalid_response",
            ) from e

    def _post(
        self,
        path: str,
        json_body: Optional[Union[dict, list]] = None,
        *,
        error_label: str = "запрос",
        error_code: str = "siu_error",
    ) -> Any:
        """POST api_base + path с телом json_body. Возвращает response.json(). При ошибке — ServiceError."""
        url = self._api_base + path
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    cookies=self._cookies,
                    json=json_body,
                    headers={"Content-Type": "application/json;charset=utf-8"},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise ServiceError(
                error=f"Ошибка сервиса СИУ ({error_label})",
                detail=f"Сервис вернул {e.response.status_code}: {e.response.text[:200] if e.response.text else ''}",
                status_code=e.response.status_code,
                code=error_code,
            ) from e
        except httpx.HTTPError as e:
            raise ServiceError(
                error=f"Ошибка соединения с сервисом СИУ ({error_label})",
                detail=str(e),
                status_code=503,
                code="siu_connection_error",
            ) from e
        except json.JSONDecodeError as e:
            raise ServiceError(
                error=f"Ошибка ответа сервиса СИУ ({error_label})",
                detail=f"Некорректный JSON в ответе: {e}",
                status_code=502,
                code="siu_invalid_response",
            ) from e

    def get_current_user_info(self) -> dict[str, Any]:
        """Запрос информации о текущем пользователе (user/current)."""
        return self._get(
            "/user/current",
            error_label="получение данных текущего пользователя",
            error_code="user_service_error",
        )

    def get_irv_info(self, irv_id: str) -> dict[str, Any]:
        """Запрос информации о версии информационного объекта (краткие данные по irv_id)."""
        return self._get(
            f"/irv/{irv_id}",
            error_label="получение данных версии информационного объекта",
            error_code="irv_service_error",
        )

    def get_nau_tir_ids(self, nau_id: str) -> dict[str, str]:
        """Список ИД типов ИР по NAU (reportSiuNauTirIds). Возвращает dict: name -> id."""
        raw = self._get(
            f"/nau/{nau_id}/tirs",
            error_label="получение списка типов ИР по NAU",
            error_code="nau_tirs_service_error",
        )
        items = raw if isinstance(raw, list) else (raw.get("contents", raw) if isinstance(raw, dict) else [])
        tir_ids: dict[str, str] = {}
        for tir in items:
            data = tir.get("data", tir) if isinstance(tir, dict) else tir
            if isinstance(data, dict) and "name" in data and "id" in data:
                tir_ids[data["name"]] = data["id"]
        return tir_ids

    def get_tir_metas(
        self,
        tir_id: str,
        *,
        depth: int = 1,
        with_base_metas: bool = False,
        with_dict_childs: bool = True,
        with_dict_childs_as_object: bool = True,
    ) -> dict[str, Any]:
        """Метаданные типа ИР (reportSiuTirMetas). Возвращает dict: name -> meta."""
        body = {
            "depth": depth,
            "withBaseMetas": with_base_metas,
            "withDictChilds": with_dict_childs,
            "withDictChildsAsObject": with_dict_childs_as_object,
        }
        raw = self._post(
            f"/tir/{tir_id}/metas",
            json_body=body,
            error_label="получение метаданных типа ИР",
            error_code="tir_metas_service_error",
        )
        items = raw if isinstance(raw, list) else (raw.get("contents", raw) if isinstance(raw, dict) else [])
        meta_ids: dict[str, Any] = {}
        for meta in items:
            data = meta.get("data", meta) if isinstance(meta, dict) else meta
            if isinstance(data, dict) and "name" in data:
                meta_ids[data["name"]] = data
        return meta_ids

    def create_folder(self, folder_name: str, parent_folder_id: str, description: Optional[str] = None) -> Any:
        """Поиск или создание папки (reportSiuCreateFolder). Ищет по имени, при отсутствии создаёт."""
        find_body = {"name": folder_name}
        old = self._post(
            f"/folder/{parent_folder_id}/childs/find",
            json_body=find_body,
            error_label="поиск дочерней папки",
            error_code="folder_find_service_error",
        )
        old_data = old.get("data", old) if isinstance(old, dict) else old
        if isinstance(old_data, dict) and old_data.get("error", "").find("not found") == -1:
            return old
        create_body = {"name": folder_name}
        if description:
            create_body["description"] = description
        new = self._post(
            f"/folder/{parent_folder_id}/childs",
            json_body=create_body,
            error_label="создание дочерней папки",
            error_code="folder_create_service_error",
        )
        return new

    def get_nau_folders(self, nau_id: str) -> Any:
        """Список папок NAU (reportSiuNauFolders)."""
        return self._get(
            f"/nau/{nau_id}/folders",
            error_label="получение списка папок NAU",
            error_code="nau_folders_service_error",
        )

    def get_folder_children(self, folder_id: str) -> Any:
        """Список дочерних элементов папки (reportSiuNauChilds / folder/childs)."""
        return self._get(
            f"/folder/{folder_id}/childs",
            error_label="получение списка дочерних папок заданной папки",
            error_code="folder_children_service_error",
        )

    def get_folder_irvs(self, folder_id: str) -> Any:
        """Список ИР в папке (reportSiuFolderIrvs)."""
        return self._get(
            f"/folder/{folder_id}/irvs",
            error_label="получение списка информационных объектов в заданной папке",
            error_code="folder_irvs_service_error",
        )

    def get_irv(
        self,
        irv_id: str,
        *,
        attr_key: str = "name",
        with_meta: bool = True,
        with_base_metas: bool = True,
        with_semantic: bool = True,
        with_files: bool = True,
        plane_values: bool = False,
        with_dict_childs: bool = False,
        with_dict_childs_as_object: bool = False,
        depth: int = 0,
        dict_sort_order: str = "name",
    ) -> Any:
        """Полные данные ИР по id (reportSiuIrv). POST с телом параметров."""
        body = {
            "attrKey": attr_key,
            "withMeta": with_meta,
            "withBaseMetas": with_base_metas,
            "withSemantic": with_semantic,
            "withFiles": with_files,
            "planeValues": plane_values,
            "withDictChilds": with_dict_childs,
            "withDictChildsAsObject": with_dict_childs_as_object,
            "depth": depth,
            "dictSortOrder": dict_sort_order,
        }
        return self._post(
            f"/irv/{irv_id}",
            json_body=body,
            error_label="получение полных данных ИР",
            error_code="irv_full_service_error",
        )

    def get_irv_files(self, irv_id: str) -> Any:
        """Список файлов ИР по id (reportSiuIrvFiles по id)."""
        return self._get(
            f"/irv/{irv_id}/files",
            error_label="получение списка файлов, приложенных к информационному объекту",
            error_code="irv_files_service_error",
        )

    def post_irv_file_content(
        self,
        file: dict[str, Any],
        body: Union[bytes, str],
    ) -> Any:
        """Запись содержимого файла ИР (reportSiuPostIrvFileContent). file: dict с irvfId, name."""
        irvf_id = file.get("irvfId") if isinstance(file, dict) else None
        file_name = file.get("name", "") if isinstance(file, dict) else ""
        if not irvf_id:
            raise ServiceError(
                error="Неверный объект файла",
                detail="В объекте файла отсутствует irvfId.",
                status_code=400,
                code="invalid_file_object",
            )
        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
            content_type = "plain/text;charset=utf-8"
        else:
            body_bytes = body
            content_type = "application/octet-stream"
        crc = hashlib.md5(body_bytes).hexdigest()
        path = f"/file/{irvf_id}/write?fileName={quote(file_name, safe='')}&crc={crc}"
        url = self._api_base + path
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    cookies=self._cookies,
                    content=body_bytes,
                    headers={"Content-Type": content_type},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise ServiceError(
                error="Ошибка сервиса СИУ (запись содержимого файла ИР)",
                detail=f"Сервис вернул {e.response.status_code}: {e.response.text[:200] if e.response.text else ''}",
                status_code=e.response.status_code,
                code="irv_file_write_error",
            ) from e
        except httpx.HTTPError as e:
            raise ServiceError(
                error="Ошибка соединения с сервисом СИУ (запись файла ИР)",
                detail=str(e),
                status_code=503,
                code="siu_connection_error",
            ) from e
        except json.JSONDecodeError as e:
            raise ServiceError(
                error="Ошибка ответа сервиса СИУ (запись файла ИР)",
                detail=f"Некорректный JSON в ответе: {e}",
                status_code=502,
                code="siu_invalid_response",
            ) from e

    def get_irv_file_content(self, file: dict[str, Any]) -> Any:
        """Чтение содержимого файла ИР (reportSiuGetIrvFileContent). file: dict с irvfId."""
        irvf_id = file.get("irvfId") if isinstance(file, dict) else None
        if not irvf_id:
            raise ServiceError(
                error="Неверный объект файла",
                detail="В объекте файла отсутствует irvfId.",
                status_code=400,
                code="invalid_file_object",
            )
        return self._get(
            f"/file/{irvf_id}/read",
            error_label="чтение содержимого файла ИР",
            error_code="irv_file_read_error",
        )

    def get_irv_file_status(self, file: dict[str, Any]) -> Any:
        """Статус загрузки файла ИР (reportSiuGetIrvFileStatus). file: dict с irvfId. Возвращает uploadStatus."""
        irvf_id = file.get("irvfId") if isinstance(file, dict) else None
        if not irvf_id:
            raise ServiceError(
                error="Неверный объект файла",
                detail="В объекте файла отсутствует irvfId.",
                status_code=400,
                code="invalid_file_object",
            )
        raw = self._get(
            f"/file/{irvf_id}/status",
            error_label="получение статуса файла ИР",
            error_code="irv_file_status_error",
        )
        if isinstance(raw, dict):
            return raw.get("uploadStatus", raw)
        return raw

    def create_ir(
        self,
        irv_name: str,
        parent_folder_id: str,
        nau_id: str,
        description: Optional[str] = None,
        comment: Optional[str] = None,
        metadata: Optional[Any] = None,
        file_name: Optional[Union[str, List[str]]] = None,
        io_id: Optional[str] = None,
    ) -> Any:
        """Создание ИР в папке (reportSiuCreateIr). metadata — объект для xml; file_name — строка или список имён.
        Если передан io_id, создаётся новая версия существующего ИО (тело с ioId без поиска по имени)."""
        if io_id:
            irv: dict[str, Any] = {
                "ioId": io_id,
                "name": irv_name,
                "description": description or irv_name,
                "nauId": nau_id,
            }
            if comment:
                irv["comment"] = comment
            if metadata is not None:
                irv["xmlMetaDataString"] = metadata
            if file_name is not None:
                irv["fileName"] = "&comma;".join(file_name) if isinstance(file_name, list) else file_name
                irv["fileNameSeparator"] = "&comma;"
            return self._post(
                f"/folder/{parent_folder_id}/irvs",
                json_body=irv,
                error_label="создание новой версии ИР",
                error_code="irv_create_service_error",
            )
        find_body = {"name": irv_name, "withMetaData": False}
        raw_irvs = self._post(
            f"/folder/{parent_folder_id}/irvs/find",
            json_body=find_body,
            error_label="поиск ИР в папке",
            error_code="irv_find_service_error",
        )
        irvs_list = raw_irvs if isinstance(raw_irvs, list) else (raw_irvs.get("contents", []) if isinstance(raw_irvs, dict) else [])
        irv = find_body.copy()
        for the_irv in irvs_list:
            data = the_irv.get("data", the_irv) if isinstance(the_irv, dict) else the_irv
            if isinstance(data, dict) and data.get("name") == irv_name:
                irv = data.copy()
                break
        irv["description"] = description or irv_name
        if comment:
            irv["comment"] = comment
        irv["nauId"] = nau_id
        if metadata is not None:
            irv["xmlMetaDataString"] = metadata  # вызывающий код может подставить XML-строку
        if "ir" in irv and irv["ir"] and isinstance(irv["ir"], dict):
            irv["ioId"] = irv["ir"].get("id")
        elif "ir" in irv and irv["ir"] and hasattr(irv["ir"], "get"):
            ir_data = irv["ir"].get("data", irv["ir"]) if isinstance(irv["ir"], dict) else irv["ir"]
            irv["ioId"] = ir_data.get("id") if isinstance(ir_data, dict) else None
        if file_name is not None:
            irv["fileName"] = "&comma;".join(file_name) if isinstance(file_name, list) else file_name
            irv["fileNameSeparator"] = "&comma;"
        return self._post(
            f"/folder/{parent_folder_id}/irvs",
            json_body=irv,
            error_label="создание ИР в папке",
            error_code="irv_create_service_error",
        )

    def build_create_meta_value(self, meta: dict[str, Any], value: Any) -> dict[str, Any]:
        """Формирует объект значения метаданных (reportSiuCreateMetaValue). Без вызова API."""
        meta_data = meta.get("data", meta) if isinstance(meta, dict) else meta
        if not isinstance(meta_data, dict):
            meta_data = meta if isinstance(meta, dict) else {}
        type_meta = meta_data.get("typeMeta")
        if isinstance(type_meta, dict):
            type_code = type_meta.get("id") or (type_meta.get("data") or {}).get("id")
        else:
            type_code = None
        tim: dict[str, Any] = {
            "timId": meta_data.get("id"),
            "name": meta_data.get("name"),
            "typeCode": type_code,
            "value": value,
        }
        return tim

    def build_create_ir_metadata_dict(
        self,
        tir_id: str,
        meta_values: List[dict[str, Any]],
    ) -> dict[str, Any]:
        """Формирует объект метаданных ИР для создания (reportSiuCreateIrMetadataDict). Без вызова API."""
        return {
            "operationGetTypeData": {
                "typeIOId": tir_id,
                "typeIOMetaList": {"typeIOMeta": meta_values},
            },
        }
