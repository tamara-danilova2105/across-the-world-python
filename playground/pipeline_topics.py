from transformers import pipeline


REVIEWS = [
    "Тур был потрясающий, гид очень интересно рассказывал.",
    "Всё было нормально, но ничего особенного.",
    "Организация ужасная, трансфер задержали на два часа.",
    "Отель хороший, завтраки вкусные, маршрут насыщенный.",
    "Мне не понравилось проживание, номер был грязный.",
    "Гид был внимательный и очень профессиональный.",
    "Еда средняя, но виды были невероятные.",
    "Поездка оставила смешанные впечатления.",
    "Очень плохо организован трансфер.",
    "Хочу поехать ещё раз, всё было супер!",
]


SHORT_LABELS = [
    "еда",
    "гид",
    "маршрут",
    "логистика",
    "проживание",
]

DESCRIPTIVE_LABELS = [
    "качество питания",
    "работа гида",
    "маршрут тура",
    "логистика и трансфер",
    "проживание и отель",
]


def print_result(title: str, result: dict, threshold: float = 0.35) -> None:
    print(f"\n{title}")
    print("-" * 80)

    print("RAW LABELS/SCORES:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label}: {round(float(score), 3)}")

    print(f"\nFILTERED >= {threshold}:")
    found = False

    for label, score in zip(result["labels"], result["scores"]):
        score = float(score)

        if score >= threshold:
            found = True
            print(f"- {label}: {round(score, 2)}")

    if not found:
        print("- ничего не прошло threshold")


def main() -> None:
    model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )

    for i, text in enumerate(REVIEWS, start=1):
        print("\n" + "=" * 100)
        print(f"#{i}")
        print(f"Текст: {text}")

        result_short_single = model(
            text,
            candidate_labels=SHORT_LABELS,
            multi_label=False,
        )

        result_short_multi = model(
            text,
            candidate_labels=SHORT_LABELS,
            multi_label=True,
        )

        result_descriptive_multi_without_template = model(
            text,
            candidate_labels=DESCRIPTIVE_LABELS,
            multi_label=True,
        )

        result_descriptive_multi_with_template = model(
            text,
            candidate_labels=DESCRIPTIVE_LABELS,
            multi_label=True,
            hypothesis_template="Этот отзыв про {}.",
        )

        print_result(
            "1. Короткие labels + multi_label=False",
            result_short_single,
            threshold=0.2,
        )

        print_result(
            "2. Короткие labels + multi_label=True",
            result_short_multi,
            threshold=0.35,
        )

        print_result(
            "3. Описательные labels + multi_label=True + без hypothesis_template",
            result_descriptive_multi_without_template,
            threshold=0.35,
        )

        print_result(
            "4. Описательные labels + multi_label=True + hypothesis_template",
            result_descriptive_multi_with_template,
            threshold=0.35,
        )


if __name__ == "__main__":
    main()