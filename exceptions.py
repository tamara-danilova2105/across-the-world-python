from __future__ import annotations

import uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from core.errors import AppError
from schemas.errors import ErrorCode, ErrorDetail, ErrorResponse


def install_exception_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        payload = ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        details = [
            ErrorDetail(
                field=".".join(str(x) for x in e.get("loc", []) if x != "body") or None,
                message=e.get("msg", "Validation error"),
            )
            for e in exc.errors()
        ]

        payload = ErrorResponse(
            code=ErrorCode.VALIDATION_ERROR,
            message="Некорректный запрос",
            details=details,
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        payload = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Внутренняя ошибка сервера",
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(mode="json"),
        )
