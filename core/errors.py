from __future__ import annotations

from typing import List, Optional

from schemas.errors import ErrorCode, ErrorDetail


class AppError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: ErrorCode,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(message)
