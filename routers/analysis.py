from fastapi import APIRouter
from db.mongo import collection
from services.sentiment import analyze_text

router = APIRouter()

@router.get(
    "/reviews/analysis",
    tags=["Отзывы"],
    summary="Анализ отзывов",
    description="Возвращает агрегированную статистику по тональности отзывов."
)
def analyze_reviews():
    reviews = list(collection.find({"isModeration": False}))  # только опубликованные

    summary = {
        "total": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "detailed": []
    }

    for r in reviews:
        feedback = r.get("feedback", "")
        sentiment = analyze_text(feedback)
        label = sentiment["label"].lower()

        summary["total"] += 1
        if label in summary:
            summary[label] += 1

        summary["detailed"].append({
            "_id": str(r["_id"]),
            "name": r.get("name", "аноним"),
            "city": r.get("city", ""),
            "text": feedback,
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
            "createdAt": r.get("createdAt")
        })

    return summary
