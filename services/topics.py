import logging
from functools import lru_cache

from schemas.reviews import TopicScore
from services.topic_labels import TOPIC_LABELS, get_topic_code_by_label

logger = logging.getLogger(__name__)


TOPIC_THRESHOLD = 0.3
MAX_TOPICS = 3


@lru_cache(maxsize=1)
def get_topic_model():
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


def detect_topics(text: str) -> list[TopicScore]:
    result = get_topic_model()(
            text, 
            candidate_labels=list(TOPIC_LABELS.values()),
            multi_label=True,
            hypothesis_template="Этот отзыв про {}.",
    )

    topics: list[TopicScore] = []

    for label, score in zip(result.get('labels', []), result.get('scores', [])):
        code = get_topic_code_by_label(label)

        if not code:
            logger.warning("Unknown topic label from model: %s", label)
            continue

        topics.append(
            TopicScore(
                code=code,
                topic=label,
                score=round(float(score), 2)
            )
        )

    return topics
    
    
def rank_topics(
        topics: list[TopicScore],
        threshold: float = TOPIC_THRESHOLD,
        max_topics: int = MAX_TOPICS,
) -> list[TopicScore]:
    filtered = [
        topic for topic in topics
        if topic.score >= threshold
    ]

    return sorted(
        filtered,
        key=lambda topic: topic.score,
        reverse=True
    )[:max_topics]
    

def analyze_topics(text: str) -> list[TopicScore]:
    if not text or not text.strip():
        return []

    try:
        detected_topics = detect_topics(text)
        return rank_topics(detected_topics)
    except Exception:
        logger.exception("Topic model failed")
        return []  # темы не должны валить весь анализ

