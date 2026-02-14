from transformers import pipeline
from starlette import status

from core.errors import AppError
from schemas.errors import ErrorCode, ErrorDetail

# инициализация модели. создаёт NLP pipeline из HuggingFace
# объект, который: принимает текст, возвращает sentiment
sentiment_model = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# допустимые лейблы от модели. если модель вернёт что-то другое - это будет ошибкой
_ALLOWED = {"positive", "neutral", "negative"}

def analyze_text(text: str) -> dict:
    # 1) бизнес-валидация входа. проверяет: text = None, text = "", text = "   "
    if not text or not text.strip():
        # если текст пустой - это не ошибка NLP-модели, а ошибка данных от клиента. возвращаем 400, а не 502
        raise AppError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=ErrorCode.VALIDATION_ERROR,
            message="Текст отзыва пустой",
            details=[ErrorDetail(field="feedback", message="Обязательное поле")],
        )

    # 2) защита от падений/неожиданностей модели
    try:
        result = sentiment_model(text)[0]
    # обработка ошибки модели. например, если модель недоступна, или вернула не JSON, или JSON без нужных полей    
    except Exception as exc:
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="Ошибка NLP-модели тональности",
        ) from exc

    # извлечение и базовая валидация результата модели
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))

    # 3) защита от новых/битых значений
    if label not in _ALLOWED:
        raise AppError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ErrorCode.UPSTREAM_ERROR,
            message="NLP-модель вернула неизвестную тональность",
            details=[ErrorDetail(field="sentiment.label", message=f"Unknown label: {label}")],
        )

    return {
        "label": label,             
        "score": round(score, 2),
    }
