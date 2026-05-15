# NLP Evaluation Notes

## Цель

Ручная evaluation sentiment/topic pipelines для понимания:

- как ведёт себя HuggingFace pipeline
- где модель ошибается
- какие кейсы плохо покрываются general-purpose NLP-моделью
- какие business rules нужны для production-системы

---

# Sentiment Analysis Evaluation

| # | Text | Expected sentiment | Model sentiment | Expected topics | Model topics | Comment |
|---|---|---|---|---|---|---|
| 1 | Тур был потрясающий, гид очень интересно рассказывал. | positive | neutral | работа гида | работа гида | Модель не уловила позитивный контекст |
| 2 | Всё было нормально, но ничего особенного. | neutral | neutral | - | - | OK |
| 3 | Организация ужасная, трансфер задержали на два часа. | negative | negative | логистика и трансфер | логистика и трансфер | OK |
| 4 | Отель хороший, завтраки вкусные, маршрут насыщенный. | positive | positive | проживание, питание, маршрут | проживание, маршрут | Не выделила тему питания |
| 5 | Мне не понравилось проживание, номер был грязный. | negative | neutral | проживание | проживание | Пропущен явный негатив |
| 6 | Гид был внимательный и очень профессиональный. | positive | positive | работа гида | работа гида | OK |
| 7 | Еда средняя, но виды были невероятные. | neutral / positive | neutral | питание, маршрут | питание | Mixed sentiment обработан частично |
| 8 | Поездка оставила смешанные впечатления. | neutral | positive | - | - | Ошибка интерпретации mixed sentiment |
| 9 | Очень плохо организован трансфер. | negative | neutral | логистика и трансфер | логистика | Не распознан организационный негатив |
| 10 | Хочу поехать ещё раз, всё было супер! | positive | positive | - | - | OK |

---

# Основные выводы

## 1. General-purpose модель плохо понимает domain-specific негатив

Примеры:

- "Очень плохо организован трансфер"
- "Трансфер задержали"
- "Номер был грязный"

Для человека это очевидный негатив.

Однако модель часто классифицирует такие тексты как:

```text
neutral
```

Причина:
- модель не обучалась специально на туристических отзывах
- отсутствует domain-specific knowledge

---

## 2. Mixed sentiment работает нестабильно

Пример:

```text
"Еда средняя, но виды были невероятные"
```

Модель одновременно видит:
- positive signal
- neutral/negative signal

и пытается усреднить prediction.

---

## 3. Topic detection работает лучше sentiment analysis

Zero-shot topics pipeline в большинстве случаев:
- корректно определяет тему
- хорошо работает с domain labels
- устойчивее к ambiguity

Однако:
- часть тем может теряться из-за threshold
- короткие labels работают хуже описательных

---

## 4. Описательные labels дают лучший результат

Лучше работают:

```python
[
    "качество питания",
    "работа гида",
    "логистика и трансфер",
]
```

чем:

```python
[
    "еда",
    "гид",
    "трансфер",
]
```

Причина:
- zero-shot classification фактически проверяет semantic hypothesis

Например:

```text
"Этот отзыв про логистику и трансфер"
```

---

## 5. Multi-label topics работают лучше single-label

Для отзывов:

```text
"Отель хороший, завтраки вкусные, маршрут насыщенный"
```

single-label:
- теряет часть информации

multi-label:
- позволяет выделять несколько тем одновременно

---

# Engineering выводы

## Pipeline — это abstraction

HuggingFace pipeline скрывает:

```text
text
→ tokenizer
→ input_ids
→ transformer model
→ logits
→ softmax
→ label + score
```

---

## Score ≠ истина

Высокий confidence score:
- не гарантирует корректность prediction
- отражает только уверенность модели в выбранном классе

---

## NLP-система требует evaluation

Даже хорошие HuggingFace модели:
- ошибаются
- имеют dataset bias
- плохо работают с domain language
- нестабильны на mixed sentiment

---

## Production NLP требует hybrid architecture

На практике production-системы используют:

```text
ML model
+ business rules
+ preprocessing
+ post-processing
+ evaluation datasets
```

а не только "чистую" модель.
