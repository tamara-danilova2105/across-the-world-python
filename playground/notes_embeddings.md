# Embeddings Notes

# Что такое embeddings

Embeddings — это vector representation текста.

Модель преобразует текст:

```text
"Отель был ужасный"
```

в:

```text
[0.126, -0.059, 0.353, ...]
```

Embedding хранит:
- semantic meaning
- context
- relationships between words

---

# Главная идея embeddings

Похожие по смыслу тексты должны иметь:
- близкие embeddings
- высокий cosine similarity

Например:

```text
"Отель был ужасный"
```

и:

```text
"Номер был грязный"
```

должны быть semantic close,
даже если exact words отличаются.

---

# Размерность embeddings

В проекте используется embedding размерности:

```text
384 dimensions
```

Это означает:
- embedding содержит 384 числа
- каждая координата хранит часть semantic information

Embedding можно представить как:

```text
координаты смысла текста
```

---

# Используемая embedding model

```python
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

# Почему выбрана именно эта модель

Модель:
- multilingual
- обучена для semantic similarity
- lightweight
- хорошо работает на CPU
- подходит для retrieval и RAG

---

# Что такое semantic search

Semantic search ищет:
- не exact слова
- а semantic similarity

---

# Keyword search vs Semantic search

## Keyword search

Ищет:
- exact words
- exact phrase matching

Пример:

```text
QUERY:
грязный отель
```

Найдёт:

```text
Отель был ужасный, номер грязный
```

Но может пропустить:

```text
Номер оказался неубранным
```

потому что:
- нет exact слова "отель"

---

## Semantic search

Semantic search:
- использует embeddings
- сравнивает vectors
- понимает похожий смысл

Поэтому может найти:

```text
Номер оказался неубранным,
постельное белье было грязное
```

даже без exact overlap.

---

# Semantic Search Pipeline

```text
query
→ embedding(query)
→ embeddings(documents)
→ cosine similarity
→ ranking
→ top results
```

---

# Cosine Similarity

Cosine similarity измеряет:
- насколько embeddings близки по смыслу

---

## Интерпретация

```text
1.0
→ почти одинаковый смысл
```

```text
0.0
→ почти нет semantic связи
```

---

# Edge Cases

Для проверки качества embeddings были протестированы edge cases.

---

# Negative slang

```text
отель кринж
↔
отель ужасный
```

Similarity:

```text
0.6319
```

Вывод:
- негативный slang embeddings понимают относительно хорошо

---

# Positive slang

```text
тур имба
↔
тур отличный
```

Similarity:

```text
0.4129
```

Вывод:
- позитивный slang модель понимает хуже
- modern slang может отсутствовать в train dataset

---

# Multilingual similarity

```text
Hotel was terrible
↔
Отель был ужасный
```

Similarity:

```text
0.9685
```

Вывод:
- multilingual embeddings работают очень хорошо
- модель успешно связывает русский и английский смысл

---

# Factual / neutral similarity

```text
Тур проходил в июне
↔
Поездка была в мае
```

Similarity:

```text
0.7486
```

Вывод:
- embeddings видят semantic similarity
- оба текста относятся к поездке и времени

---

# Mixed sentiment

```text
Гид супер, но отель ужасный
↔
Понравился экскурсовод, но проживание было плохим
```

Similarity:

```text
0.5071
```

Вывод:
- mixed sentiment embeddings понимают частично
- semantic overlap присутствует,
  но phrasing сильно влияет на similarity

---

# Domain logistics problem

```text
Трансфер задержали на два часа
↔
Автобус приехал с большим опозданием
```

Similarity:

```text
0.2983
```

Вывод:
- embeddings плохо уловили domain-specific связь
- "трансфер" и "автобус" для модели недостаточно близки

Это показывает:
- embeddings не идеальны
- domain-specific evaluation обязательна

---

# Главный engineering вывод

Embeddings:
- намного лучше keyword search
- умеют semantic retrieval
- являются фундаментом для:
  - RAG
  - vector DB
  - AI search systems
  - recommendation systems

Но embeddings:
- не являются "магией"
- требуют evaluation
- могут ошибаться на domain-specific кейсах
- чувствительны к slang и phrasing

---

# Что дальше

Следующие шаги:
- вынести embeddings в services/embeddings.py
- сделать /reviews/search
- добавить vector retrieval
- построить mini-RAG pipeline
