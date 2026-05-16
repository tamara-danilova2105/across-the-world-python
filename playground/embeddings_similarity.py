from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Embedding model
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Тестовые тексты
TEXT_1 = "Отель был ужасный"
TEXT_2 = "Номер был грязный"
TEXT_3 = "Гид рассказывал интересно"


def main() -> None:
    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================

    print("\n=== LOAD EMBEDDING MODEL ===")

    model = SentenceTransformer(MODEL_NAME)

    # =========================================================================
    # 2. CREATE EMBEDDINGS
    # =========================================================================

    print("\n=== CREATE EMBEDDINGS ===")

    embedding_1 = model.encode(TEXT_1)
    embedding_2 = model.encode(TEXT_2)
    embedding_3 = model.encode(TEXT_3)

    # =========================================================================
    # 3. COSINE SIMILARITY
    # =========================================================================

    print("\n=== COSINE SIMILARITY ===")

    # cosine similarity:
    # насколько embeddings похожи друг на друга
    #
    # 1.0
    # -> почти одинаковый смысл
    #
    # 0.0
    # -> почти нет связи

    similarity_1_2 = cosine_similarity(
        [embedding_1],
        [embedding_2],
    )[0][0]

    similarity_1_3 = cosine_similarity(
        [embedding_1],
        [embedding_3],
    )[0][0]

    # =========================================================================
    # 4. RESULTS
    # =========================================================================

    print("\nTEXT 1:")
    print(TEXT_1)

    print("\nTEXT 2:")
    print(TEXT_2)

    print("\nTEXT 3:")
    print(TEXT_3)

    print("\n" + "=" * 100)

    print("\nSIMILARITY:")
    print(f"{TEXT_1} <-> {TEXT_2}")
    print(round(float(similarity_1_2), 4))

    print("\nSIMILARITY:")
    print(f"{TEXT_1} <-> {TEXT_3}")
    print(round(float(similarity_1_3), 4))

    # =========================================================================
    # 5. SIMPLE INTERPRETATION
    # =========================================================================

    print("\n=== INTERPRETATION ===")

    if similarity_1_2 > similarity_1_3:
        print(
            f'"{TEXT_1}" semantic closer to "{TEXT_2}"'
        )
    else:
        print(
            f'"{TEXT_1}" semantic closer to "{TEXT_3}"'
        )


if __name__ == "__main__":
    main()