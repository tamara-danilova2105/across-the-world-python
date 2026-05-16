import logging
from typing import Optional

from db.mongo import collection
from schemas.reviews import SentimentLabel
from services.embeddings import create_embedding, calculate_similarity
from services.sentiment import analyze_text
from services.topics import analyze_topics

logger = logging.getLogger(__name__)


def search_reviews_service(
    query: str,
    limit: int = 5,
    min_score: float = 0.25,
    sentiment_filter: Optional[SentimentLabel] = None,
    topic_filter: Optional[str] = None,
) -> dict:
    reviews = collection.find({"isModeration": True})

    query_embedding = create_embedding(query)

    results = []

    for r in reviews:
        feedback = str(r.get("feedback") or "").strip()

        if not feedback:
            continue

        try:
            review_embedding = create_embedding(feedback)

            similarity_score = calculate_similarity(
                query_embedding,
                review_embedding,
            )

            if similarity_score < min_score:
                continue

            sentiment = analyze_text(feedback)
            sentiment_label = SentimentLabel(sentiment["label"])

            if sentiment_filter and sentiment_label != sentiment_filter:
                continue

            topics = analyze_topics(feedback)

            if topic_filter:
                topic_filter_lower = topic_filter.lower()

                has_topic = any(
                    topic_filter_lower in topic.topic.lower()
                    for topic in topics
                )

                if not has_topic:
                    continue

        except Exception:
            logger.exception(
                "Hybrid search failed for review: %s",
                r.get("_id"),
            )
            continue

        results.append(
            {
                "id": str(r["_id"]),
                "name": r.get("name", "аноним"),
                "city": r.get("city"),
                "text": feedback,
                "score": round(float(similarity_score), 4),
                "sentiment": sentiment_label,
                "sentimentScore": sentiment["score"],
                "topics": topics,
                "createdAt": r.get("createdAt"),
            }
        )

    results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": query,
        "total": len(results),
        "results": results[:limit],
    }