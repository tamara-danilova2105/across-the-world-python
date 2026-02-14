import logging
from functools import lru_cache

from schemas.reviews import TopicScore

logger = logging.getLogger(__name__)

LABELS = ["еда", "гид", "маршрут", "логистика", "проживание"]


@lru_cache(maxsize=1)
def get_topic_model():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


def analyze_topics(text: str) -> list[TopicScore]:
    if not text or not text.strip():
        return []

    try:
        result = get_topic_model()(text, candidate_labels=LABELS)
    except Exception:
        logger.exception("Topic model failed")
        return []  # темы не должны валить весь анализ

    out: list[TopicScore] = []
    for label, score in zip(result.get("labels", []), result.get("scores", [])):
        score_f = float(score)
        if score_f > 0.2:
            out.append(TopicScore(topic=label, score=round(score_f, 2)))

    return out
