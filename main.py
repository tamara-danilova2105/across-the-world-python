from fastapi import FastAPI
from routers.analysis import router as analysis_router
from exceptions import install_exception_handlers

app = FastAPI(title="Review Analysis API", version="1.0.0")
install_exception_handlers(app)


app.include_router(analysis_router, prefix="/v1")

