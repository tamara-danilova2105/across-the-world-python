from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Загружает embedding model один раз.

    SentenceTransformer тяжелый,
    поэтому модель нельзя создавать на каждый запрос.
    """

    return SentenceTransformer(MODEL_NAME)


def create_embedding(text: str) -> list[float]:
    """
    Создаёт embedding для одного текста.
    """

    model = get_embedding_model()

    embedding = model.encode(text)

    return embedding.tolist()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Создаёт embeddings для списка текстов.
    """

    model = get_embedding_model()

    embeddings = model.encode(texts)

    return embeddings.tolist()


def calculate_similarity(
    embedding_1: list[float],
    embedding_2: list[float],
) -> float:
    """
    Считает cosine similarity между embeddings.
    """

    similarity = cosine_similarity(
        [embedding_1],
        [embedding_2],
    )[0][0]

    return float(similarity)