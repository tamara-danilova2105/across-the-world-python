from typing import Any, Dict


NEGATIVE_MARKERS = [
    "не понравилось",
    "не понравился",
    "не понравилась",
    "грязный",
    "грязная",
    "грязное",
    "грязно",
    "плохо",
    "ужасно",
    "ужасная",
    "ужасный",
    "ужасное",
    "задержали",
    "задержка",
    "опоздал",
    "опоздала",
    "опоздали",
    "кринж",
    "треш",
    "жесть",
    "такое себе",
    "токсик",
]

POSITIVE_MARKERS = [
    "имба",
    "топ",
    "вайбовый",
    "пушка",
    "кайф",
    "огонь",
    "чиллово",
]


def has_marker(text: str, markers: list[str]) -> bool:
    text_lower = text.lower()

    return any(marker in text_lower for marker in markers)


def apply_business_rules(text: str, sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Усиливает результат модели доменными правилами.

    Hybrid-подход:
    HuggingFace model + domain/slang rules.
    """
    has_negative_marker = has_marker(text, NEGATIVE_MARKERS)
    has_positive_marker = has_marker(text, POSITIVE_MARKERS)

    if has_negative_marker:
        return {
            "label": "negative",
            "score": max(float(sentiment["score"]), 0.8),
        }

    if has_positive_marker:
        return {
            "label": "positive",
            "score": max(float(sentiment["score"]), 0.8),
        }

    return sentiment