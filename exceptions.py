from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from core.errors import AppError
from schemas.errors import ErrorCode, ErrorResponse

def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        payload = ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=getattr(exc, "details", []) or [],
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        payload = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Внутренняя ошибка сервера",
            details=[],
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
