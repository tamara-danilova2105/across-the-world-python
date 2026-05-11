from transformers import pipeline

from services.sentiment import analyze_text


EDGE_CASES = [
    {
        "text": "Ну просто великолепный трансфер, автобус опоздал на 6 часов",
        "expected": "negative",
        "problem": "sarcasm",
    },
    {
        "text": "Норм",
        "expected": "neutral",
        "problem": "too short",
    },
    {
        "text": "Тур проходил в июне",
        "expected": "neutral",
        "problem": "factual / no sentiment",
    },
    {
        "text": "Гид супер, но отель ужасный",
        "expected": "negative",
        "problem": "mixed sentiment",
    },
    {
        "text": "Тур имба, гид вообще топ",
        "expected": "positive",
        "problem": "youth slang",
    },
    {
        "text": "Отель кринж, номер вообще жесть",
        "expected": "negative",
        "problem": "youth slang",
    },
    {
        "text": "Маршрут вайбовый, но трансфер кринж",
        "expected": "negative",
        "problem": "slang + mixed sentiment",
    },
    {
        "text": "Еда норм, отель такое себе",
        "expected": "negative",
        "problem": "colloquial vague",
    },
]


def main() -> None:
    raw_model = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
    )

    for case in EDGE_CASES:
        raw = raw_model(case["text"])[0]
        raw_label = raw["label"].lower()
        raw_score = round(float(raw["score"]), 2)

        app_result = analyze_text(case["text"])
        app_label = app_result["label"]
        app_score = app_result["score"]

        print("\n" + "=" * 100)
        print(f"TEXT: {case['text']}")
        print(f"EXPECTED: {case['expected']}")
        print(f"PROBLEM: {case['problem']}")

        print("\nRAW MODEL:")
        print(f"- label: {raw_label}")
        print(f"- score: {raw_score}")
        print(f"- match: {raw_label == case['expected']}")

        print("\nAPP RESULT:")
        print(f"- label: {app_label}")
        print(f"- score: {app_score}")
        print(f"- match: {app_label == case['expected']}")

        if raw_label != app_label:
            print("\nRULES EFFECT:")
            print(f"- changed: {raw_label} → {app_label}")
        else:
            print("\nRULES EFFECT:")
            print("- no change")


if __name__ == "__main__":
    main()