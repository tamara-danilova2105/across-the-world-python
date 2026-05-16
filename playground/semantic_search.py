from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

QUERY = "грязный отель"

REVIEWS = [
    "Отель был ужасный, номер грязный.",
    "Гид рассказывал очень интересно.",
    "Трансфер задержали на два часа.",
    "Завтраки были вкусные, ресторан понравился.",
    "Номер оказался неубранным, постельное белье было грязное.",
    "Маршрут был насыщенный и красивый.",
]


def keyword_search(
    query: str,
    reviews: list[str],
) -> list[str]:
    """
    Простейший keyword search.

    Ищет exact совпадения слов.
    Не понимает semantic meaning.
    """

    query_words = query.lower().split()

    results = []

    for review in reviews:
        review_lower = review.lower()

        # Если хотя бы одно слово из query найдено
        # в review -> добавляем результат
        if any(word in review_lower for word in query_words):
            results.append(review)

    return results


def main() -> None:
    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================

    print("\n=== LOAD EMBEDDING MODEL ===")

    model = SentenceTransformer(MODEL_NAME)

    # =========================================================================
    # 2. KEYWORD SEARCH
    # =========================================================================

    print("\n" + "=" * 100)
    print("KEYWORD SEARCH")
    print("=" * 100)

    keyword_results = keyword_search(
        QUERY,
        REVIEWS,
    )

    print(f"\nQUERY: {QUERY}")

    print("\nRESULTS:")

    if not keyword_results:
        print("- ничего не найдено")

    else:
        for review in keyword_results:
            print(f"- {review}")

    # =========================================================================
    # 3. CREATE QUERY EMBEDDING
    # =========================================================================

    print("\n=== CREATE QUERY EMBEDDING ===")

    query_embedding = model.encode(QUERY)

    # =========================================================================
    # 4. CREATE REVIEW EMBEDDINGS
    # =========================================================================

    print("\n=== CREATE REVIEW EMBEDDINGS ===")

    review_embeddings = model.encode(REVIEWS)

    # =========================================================================
    # 5. SEMANTIC SEARCH
    # =========================================================================

    print("\n" + "=" * 100)
    print("SEMANTIC SEARCH")
    print("=" * 100)

    results = []

    for review, review_embedding in zip(
        REVIEWS,
        review_embeddings,
    ):
        # Cosine similarity:
        # насколько embeddings близки по смыслу
        similarity = cosine_similarity(
            [query_embedding],
            [review_embedding],
        )[0][0]

        results.append(
            {
                "review": review,
                "score": round(float(similarity), 4),
            }
        )

    # Сортируем:
    # самый близкий semantic match -> сверху
    results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

    print(f"\nQUERY: {QUERY}")

    print("\nRESULTS:")

    for item in results:
        print("\n" + "-" * 80)

        print(f"score: {item['score']}")
        print(f"review: {item['review']}")

    # =========================================================================
    # 6. INTERPRETATION
    # =========================================================================

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        """
Keyword search:
- ищет exact слова
- не понимает смысл

Semantic search:
- ищет semantic similarity
- может находить похожие отзывы
  даже без одинаковых слов
"""
    )


if __name__ == "__main__":
    main()