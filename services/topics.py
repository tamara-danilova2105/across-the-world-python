from transformers import pipeline

# Загружаем мульти-язычную zero-shot модель
topic_model = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# Темы на русском языке
LABELS = ["еда", "гид", "маршрут", "логистика", "проживание"]

def analyze_topics(text: str) -> list[str]:
    result = topic_model(text, candidate_labels=LABELS)
    return [
        label
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.2
    ]
