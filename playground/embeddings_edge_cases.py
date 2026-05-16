from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PAIRS = [
    {
        "text_1": "отель кринж",
        "text_2": "отель ужасный",
        "expected": "high",
        "case": "negative slang",
    },
    {
        "text_1": "тур имба",
        "text_2": "тур отличный",
        "expected": "high",
        "case": "positive slang",
    },
    {
        "text_1": "Hotel was terrible",
        "text_2": "Отель был ужасный",
        "expected": "high",
        "case": "multilingual",
    },
    {
        "text_1": "Тур проходил в июне",
        "text_2": "Поездка была в мае",
        "expected": "medium/high",
        "case": "factual / neutral",
    },
    {
        "text_1": "Гид супер, но отель ужасный",
        "text_2": "Понравился экскурсовод, но проживание было плохим",
        "expected": "high",
        "case": "mixed sentiment",
    },
    {
        "text_1": "Трансфер задержали на два часа",
        "text_2": "Автобус приехал с большим опозданием",
        "expected": "high",
        "case": "domain logistics",
    },
]


def main() -> None:
    print("\n=== LOAD EMBEDDING MODEL ===")
    model = SentenceTransformer(MODEL_NAME)

    print("\n=== EMBEDDINGS EDGE CASES ===")

    for pair in PAIRS:
        embedding_1 = model.encode(pair["text_1"])
        embedding_2 = model.encode(pair["text_2"])

        similarity = cosine_similarity(
            [embedding_1],
            [embedding_2],
        )[0][0]

        print("\n" + "=" * 100)
        print(f"CASE: {pair['case']}")
        print(f"EXPECTED: {pair['expected']}")

        print("\nTEXT 1:")
        print(pair["text_1"])

        print("\nTEXT 2:")
        print(pair["text_2"])

        print("\nCOSINE SIMILARITY:")
        print(round(float(similarity), 4))


if __name__ == "__main__":
    main()