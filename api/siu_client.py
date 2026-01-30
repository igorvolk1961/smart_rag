"""
Клиент REST API СИУ (единый базовый URL, аутентификация по JSESSIONID).
Содержит методы для запросов к сервисам СИУ; при необходимости добавляются новые GET/POST.
"""

import json
from typing import Any, Optional

import httpx

from api.exceptions import ServiceError


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
        self._cookies = {"JSESSIONID": jsessionid.strip()}
        self._timeout = timeout

    def _get(
        self,
        path: str,
        *,
        error_label: str = "запрос",
        error_code: str = "siu_error",
    ) -> dict[str, Any]:
        """GET base_url + path, возвращает response.json(). При ошибке — ServiceError."""
        url = self._base_url + path
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

    def get_current_user_info(self) -> dict[str, Any]:
        """Запрос информации о текущем пользователе."""
        return self._get(
            "/siu-star/services/api/user/current",
            error_label="получение данных текущего пользователя",
            error_code="user_service_error",
        )

    def get_irv_info(self, irv_id: str) -> dict[str, Any]:
        """Запрос информации о версии информационного объекта."""
        return self._get(
            f"/siu-star/services/api/irv/{irv_id}",
            error_label="получение данных версии информационного объекта",
            error_code="irv_service_error",
        )
