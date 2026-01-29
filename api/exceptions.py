"""
Исключения API с маппингом в HTTP-ответы.
"""

from typing import Optional


class ServiceError(Exception):
    """Ошибка сервиса с HTTP-кодом и структурированным сообщением для клиента."""

    def __init__(
        self,
        error: str,
        detail: Optional[str] = None,
        status_code: int = 500,
        code: Optional[str] = None,
    ):
        self.error = error
        self.detail = detail or error
        self.status_code = status_code
        self.code = code
        super().__init__(self.detail)
