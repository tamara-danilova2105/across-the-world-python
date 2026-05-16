from typing import Optional

from fastapi import APIRouter, Query

from schemas.errors import ErrorResponse
from schemas.reviews import (
    ReviewsAnalysisResponse,
    ReviewsSearchResponse,
    SentimentLabel,
)
from services.reviews_analysis import analyze_reviews_service
from services.reviews_search import search_reviews_service


router = APIRouter()


@router.get(
    "/reviews/analysis",
    tags=["Отзывы"],
    summary="Анализ отзывов",
    description="Возвращает агрегированную статистику по тональности отзывов и список выделенных тем.",
    response_model=ReviewsAnalysisResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
        502: {"model": ErrorResponse, "description": "Ошибка внешней NLP-модели"},
    },
)
def analyze_reviews() -> ReviewsAnalysisResponse:
    return analyze_reviews_service()


@router.get(
    "/reviews/search",
    tags=["Отзывы"],
    summary="Hybrid semantic search по отзывам",
    description="Ищет отзывы по смыслу с фильтрацией по тональности и теме.",
    response_model=ReviewsSearchResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
        502: {"model": ErrorResponse, "description": "Ошибка NLP/embedding-модели"},
    },
)
def search_reviews(
    q: str = Query(..., min_length=2, description="Поисковый запрос"),
    limit: int = Query(5, ge=1, le=20, description="Количество результатов"),
    min_score: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Минимальный similarity score",
    ),
    sentiment: Optional[SentimentLabel] = Query(
        None,
        description="Фильтр по тональности",
    ),
    topic: Optional[str] = Query(
        None,
        description="Фильтр по теме: food, guide, route, logistics, hotel",
    ),
) -> ReviewsSearchResponse:
    return search_reviews_service(
        query=q,
        limit=limit,
        min_score=min_score,
        sentiment_filter=sentiment,
        topic_filter=topic,
    )