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

def main() -> None:
    model = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
    )

    for i, text in enumerate(REVIEWS, start=1):
        raw_result = model(text)

        result = raw_result[0]
        label = result["label"].lower()
        score = round(float(result["score"]), 2)

        print("\n" + "=" * 80)
        print(f"#{i}")
        print(f"#Текст: {text}")

        print("\nRAW RESULT:")
        print(raw_result)

        print("\nNORMALIZED RESULT:")
        print(f"label: {label}")
        print(f"score: {score}")

if __name__ ==  "__main__":
    main()