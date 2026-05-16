from db.mongo import collection
from schemas.reviews import (
    ReviewsAnalysisResponse,
    ReviewsAnalysisStats,
    ReviewAnalysisItem,
    SentimentLabel,
)
from services.sentiment import analyze_text
from services.topics import analyze_topics


def analyze_reviews_service() -> ReviewsAnalysisResponse:
    reviews = collection.find({"isModeration": True})

    stats = {
        "total": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }

    detailed: list[ReviewAnalysisItem] = []

    for r in reviews:
        feedback = str(r.get("feedback") or "").strip()

        if not feedback:
            continue

        try:
            sentiment = analyze_text(feedback)
            label = SentimentLabel(sentiment["label"])
            topics = analyze_topics(feedback)
        except Exception:
            continue

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
                topics=topics,
                createdAt=r.get("createdAt"),
            )
        )

    return ReviewsAnalysisResponse(
        stats=ReviewsAnalysisStats(**stats),
        detailed=detailed,
    )