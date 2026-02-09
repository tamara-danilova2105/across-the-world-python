from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"      # внешние сервисы/модели
    INTERNAL_ERROR = "INTERNAL_ERROR"      # всё неожиданное


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Optional[str] = Field(default=None, description="Поле/путь, к которому относится ошибка")
    message: str = Field(..., min_length=1)
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Доп. данные (без персональных данных)")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str = Field(..., min_length=1)
    details: List[ErrorDetail] = Field(default_factory=list)
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
