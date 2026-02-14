import logging
from functools import lru_cache
from typing import Any, Dict

from starlette import status

from core.errors import AppError
from schemas.errors import ErrorCode, ErrorDetail

logger = logging.getLogger(__name__)

# Допустимые лейблы в ответе нашей модели/контракта
_ALLOWED = {"positive", "neutral", "negative"}


@lru_cache(maxsize=1)
def get_sentiment_model():
    """
    Инициализирует HuggingFace pipeline один раз и кэширует.
    Важно: pipeline тяжелый, нельзя создавать на каждый запрос/отзыв.
    """
    try:
        from transformers import pipeline
    except Exception as exc:
        logger.exception("Failed to import transformers")
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="Не удалось загрузить зависимости NLP-модели",
        ) from exc

    try:
        return pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment",
        )
    except Exception as exc:
        logger.exception("Failed to initialize sentiment pipeline")
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="Не удалось инициализировать NLP-модель тональности",
        ) from exc


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Возвращает dict вида:
      {"label": "positive"|"neutral"|"negative", "score": 0.0..1.0}
    """
    # 1) Валидация входа (это 400, не 502)
    if text is None or not str(text).strip():
        raise AppError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=ErrorCode.VALIDATION_ERROR,
            message="Текст отзыва пустой",
            details=[ErrorDetail(field="feedback", message="Обязательное поле")],
        )

    # 2) Вызов модели + защита
    try:
        model = get_sentiment_model()
        raw = model(text)

        # pipeline обычно возвращает list[dict]
        if isinstance(raw, list) and raw:
            result = raw[0]
        elif isinstance(raw, dict):
            result = raw
        else:
            raise ValueError(f"Unexpected pipeline result: {type(raw)}")
    except AppError:
        # уже наш контролируемый AppError — просто пробрасываем
        raise
    except Exception as exc:
        logger.exception("Sentiment model failed during inference")
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="Ошибка NLP-модели тональности",
        ) from exc

    # 3) Извлечение и базовая валидация результата
    try:
        label = str(result.get("label", "")).lower()
        score = float(result.get("score", 0.0))
    except Exception as exc:
        logger.exception("Invalid sentiment model output format: %r", result)
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="NLP-модель вернула некорректный формат ответа",
        ) from exc

    # 4) Защита от новых/битых значений
    if label not in _ALLOWED:
        logger.error("Unsupported sentiment label from model: %r (full=%r)", label, result)
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="NLP-модель вернула неизвестную тональность",
            details=[ErrorDetail(field="label", message=f"Unknown label: {label}")],
        )

    return {
        "label": label,
        "score": round(score, 2),
    }
