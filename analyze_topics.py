from transformers import pipeline

# Англоязычная zero-shot модель
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Кандидаты на темы
labels = ["food", "guide", "route", "accommodation", "logistics"]

# Пример отзыва (можно на русском — модель работает, хоть и не идеально)
text = "Очень понравился маршрут и гид, но еда была посредственная."

# Классификация
result = classifier(text, candidate_labels=labels)

# Вывод
print("Основные темы:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"{label} — {round(score, 2)}")
