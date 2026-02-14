
from schemas.reviews import TopicScore

# Загружаем мульти-язычную zero-shot модель
# pipeline создаёт объект-классификатор, который умеет определять, насколько текст относится к каждой теме из списка.
_topic_model = None

from schemas.reviews import TopicScore

_topic_model = None
LABELS = ["еда", "гид", "маршрут", "логистика", "проживание"]


def get_topic_model():
    global _topic_model
    if _topic_model is None:
        from transformers import pipeline

        _topic_model = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
    return _topic_model


def analyze_topics(text: str) -> list[TopicScore]:
    if not text or not text.strip():
        return []

    result = get_topic_model()(text, candidate_labels=LABELS)

    out: list[TopicScore] = []
    for label, score in zip(result["labels"], result["scores"]):
        score_f = float(score)
        if score_f > 0.2:
            out.append(TopicScore(topic=label, score=round(score_f, 2)))
    return out


# Темы на русском языке
LABELS = ["еда", "гид", "маршрут", "логистика", "проживание"]

def analyze_topics(text: str) -> list[TopicScore]:
    if not text or not text.strip():
        return []

    result = get_topic_model()(text, candidate_labels=LABELS)

    out: list[TopicScore] = []
    #zip — это функция Python, которая объединяет два списка попарно. 
    # В данном случае она объединяет список тем (result["labels"]) и соответствующих им оценок (result["scores"]).
    for label, score in zip(result["labels"], result["scores"]):
        score_f = float(score)
        if score_f > 0.2:
            out.append(TopicScore(topic=label, score=round(score_f, 2)))
    return out