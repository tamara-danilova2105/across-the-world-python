from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

def analyze_text(text: str) -> dict:
    result = sentiment_model(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 2)
    }
