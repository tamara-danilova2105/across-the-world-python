from fastapi import APIRouter

from db.mongo import collection
from services.sentiment import analyze_text
from services.topics import analyze_topics
from schemas.reviews import (
    ReviewsAnalysisResponse,
    ReviewsAnalysisStats,
    ReviewAnalysisItem,
    SentimentLabel,
)

from schemas.errors import ErrorResponse

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
    reviews = collection.find({"isModeration": True})

    stats = {
        "total": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }

    detailed: list[ReviewAnalysisItem] = []

    for r in reviews:
        feedback = r.get("feedback", "")
        sentiment = analyze_text(feedback)

        label = SentimentLabel(sentiment["label"])

        stats["total"] += 1
        stats[label.value] += 1

        detailed.append(
            ReviewAnalysisItem(
                id=str(r["_id"]),
                name=r.get("name", "аноним"),
                city=r.get("city"),
                text=feedback,
                sentiment=label,
                score=sentiment["score"],
                topics=analyze_topics(feedback),
                createdAt=r.get("createdAt"),
            )
        )   


    return ReviewsAnalysisResponse(
        stats=ReviewsAnalysisStats(**stats),
        detailed=detailed,
    )

