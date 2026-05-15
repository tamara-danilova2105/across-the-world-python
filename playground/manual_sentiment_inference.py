import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"


TEXTS = [
    "Тур был потрясающий, гид очень интересно рассказывал.",
    "Очень плохо организован трансфер.",
    "Тур имба, гид вообще топ",
]


def main() -> None:
    # =========================================================================
    # 1. LOAD TOKENIZER
    # =========================================================================

    print("\n=== LOAD TOKENIZER ===")

    # Tokenizer отвечает за:
    # - разбиение текста на токены
    # - преобразование текста в input_ids
    # - подготовку input для модели
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # =========================================================================
    # 2. LOAD MODEL
    # =========================================================================

    print("\n=== LOAD MODEL ===")

    # Загружаем transformer-модель для классификации текста
    #
    # AutoModelForSequenceClassification:
    # специальный тип модели для sentiment analysis / classification задач
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # =========================================================================
    # 3. MODEL LABELS
    # =========================================================================

    print("\n=== MODEL LABELS ===")

    # Показываем mapping:
    # id -> label
    #
    # Например:
    # 0 -> NEUTRAL
    # 1 -> POSITIVE
    # 2 -> NEGATIVE
    print(model.config.id2label)

    # =========================================================================
    # 4. PROCESS EACH TEXT
    # =========================================================================

    for text in TEXTS:
        print("\n" + "=" * 100)
        print(f"TEXT: {text}")

        # =====================================================================
        # 5. TOKENIZATION
        # =====================================================================

        print("\n1. TOKENIZATION")

        tokens = tokenizer.tokenize(text)

        print(f"Tokens: {tokens}")

        # =====================================================================
        # 6. ENCODE
        # =====================================================================

        print("\n2. ENCODE")

        # tokenizer(...) выполняет:
        #
        # text
        # -> tokens
        # -> input_ids
        # -> tensors
        #
        # return_tensors="pt"
        # означает:
        # вернуть PyTorch tensors
        #
        # truncation=True
        # обрезать слишком длинный текст
        #
        # padding=True
        # выровнять длину sequence при batch processing
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        # input_ids:
        # числовое представление текста
        print("\nINPUT IDS:")
        print(inputs["input_ids"])

        # attention_mask:
        # показывает:
        # 1 -> реальный token
        # 0 -> padding
        print("\nATTENTION MASK:")
        print(inputs["attention_mask"])

        # =====================================================================
        # 7. MODEL INFERENCE
        # =====================================================================

        print("\n3. MODEL INFERENCE")

        # torch.no_grad():
        # отключает вычисление gradients
        #
        # Используется для inference:
        # - быстрее
        # - меньше памяти
        # - gradients не нужны
        with torch.no_grad():

            # Передаем inputs в transformer model
            outputs = model(**inputs)

        # logits:
        # сырые предсказания модели
        logits = outputs.logits

        print("\nLOGITS:")
        print(logits)

        # =====================================================================
        # 8. SOFTMAX
        # =====================================================================

        print("\n4. SOFTMAX")

        # softmax превращает logits в probabilities
        #
        # logits:
        # [0.3, 2.8, -1.2]
        #
        # ->
        #
        # probabilities:
        # [0.07, 0.90, 0.03]
        #
        # Теперь:
        # - сумма = 1
        # - можно интерпретировать как вероятности классов
        probabilities = torch.softmax(logits, dim=1)

        print("\nPROBABILITIES:")
        print(probabilities)

        # =====================================================================
        # 9. PREDICTION
        # =====================================================================

        print("\n5. PREDICTION")

        # argmax():
        # выбирает индекс максимальной вероятности
        #
        # Например:
        # [0.07, 0.90, 0.03]
        #
        # ->
        #
        # index = 1
        predicted_class_id = probabilities.argmax().item()

        # Преобразуем class_id -> human-readable label
        #
        # Например:
        # 1 -> POSITIVE
        label = model.config.id2label[predicted_class_id]

        # Получаем confidence score выбранного класса
        score = probabilities[0][predicted_class_id].item()

        print(f"\nFINAL LABEL: {label}")
        print(f"FINAL SCORE: {round(score, 4)}")


if __name__ == "__main__":
    main()