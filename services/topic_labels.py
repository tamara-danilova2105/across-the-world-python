TOPIC_LABELS = {
    "food": "качество питания",
    "guide": "работа гида",
    "route": "маршрут тура",
    "logistics": "логистика и трансфер",
    "hotel": "проживание и отель",
}


TOPIC_CODES = set(TOPIC_LABELS.keys())


def get_topic_label(topic_code: str) -> str | None:
    return TOPIC_LABELS.get(topic_code)


def get_topic_code_by_label(label: str) -> str | None:
    for code, topic_label in TOPIC_LABELS.items():
        if topic_label == label:
            return code

    return None