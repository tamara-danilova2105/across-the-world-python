from transformers import pipeline
from schemas.reviews import TopicScore

# Загружаем мульти-язычную zero-shot модель
topic_model = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# Темы на русском языке
LABELS = ["еда", "гид", "маршрут", "логистика", "проживание"]

def analyze_topics(text: str) -> list[TopicScore]:
    if not text or not text.strip():
        return []

    result = topic_model(text, candidate_labels=LABELS)

    out: list[TopicScore] = []
    for label, score in zip(result["labels"], result["scores"]):
        score_f = float(score)
        if score_f > 0.2:
            out.append(TopicScore(topic=label, score=round(score_f, 2)))
    return out