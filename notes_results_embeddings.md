# Embeddings & Semantic Search Learning Notes

## Цель недели

На этой неделе была изучена основа modern retrieval systems:

- embeddings
- cosine similarity
- semantic search
- vector retrieval
- hybrid search

Основная задача:
понять, как работают embedding-based AI search systems под капотом.

---

# Этап 1. Первый embedding руками

Была использована модель:

```python
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Пример:

```python
embedding = model.encode(text)
```

---

## Что такое embedding

Embedding — это vector representation текста.

Например:

```text
"Отель был ужасный"
```

преобразуется в:

```text
[0.126, -0.059, 0.353, ...]
```

---

## Что хранит embedding

Embedding хранит:

- semantic meaning
- context
- relationships between words

---

## Размерность embedding

Модель создаёт vector размерности:

```text
384 dimensions
```

Это означает:
- embedding содержит 384 числа
- каждая координата кодирует часть semantic information

---

# Этап 2. Cosine similarity

Для сравнения embeddings использовался cosine similarity.

```python
cosine_similarity(
    [embedding_1],
    [embedding_2],
)
```

---

## Что измеряет cosine similarity

Cosine similarity показывает:
насколько embeddings близки по смыслу.

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

# Этап 3. Semantic search playground

Был реализован простой semantic search pipeline.

---

## Pipeline

```text
query
→ embedding(query)
→ embeddings(reviews)
→ cosine similarity
→ ranking
→ top results
```

---

## Пример

QUERY:

```text
грязный отель
```

Semantic search смог найти:

```text
Номер оказался неубранным,
постельное белье было грязное
```

даже без exact слова "отель".

---

# Этап 4. Keyword search vs Semantic search

Было проведено сравнение двух подходов.

---

## Keyword search

Ищет:
- exact words
- literal matches

---

## Ограничение keyword search

Запрос:

```text
грязный отель
```

не найдёт:

```text
номер был неубранным
```

если exact words отсутствуют.

---

## Semantic search

Semantic search:
- использует embeddings
- понимает смысл
- ищет semantic similarity

---

## Главный вывод

```text
keyword search
→ ищет слова

semantic search
→ ищет смысл
```

---

# Этап 5. Embeddings edge cases

Для проверки embeddings были протестированы edge cases.

---

## Negative slang

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

## Positive slang

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
- позитивный slang embeddings понимают хуже

---

## Multilingual similarity

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

---

## Domain-specific problem

```text
Трансфер задержали
↔
Автобус приехал с опозданием
```

Similarity:

```text
0.2983
```

Вывод:
- embeddings могут плохо понимать domain-specific связи

---

# Этап 6. services/embeddings.py

Embeddings были вынесены в отдельный service layer.

---

## Что реализовано

```python
get_embedding_model()
create_embedding()
create_embeddings()
calculate_similarity()
```

---

## Почему это важно

Такой подход:
- убирает дублирование
- позволяет переиспользовать embeddings
- делает архитектуру production-like

---

# Этап 7. /reviews/search

Был реализован semantic search endpoint:

```http
GET /reviews/search
```

---

## Что делает endpoint

```text
query
→ embedding(query)
→ embedding(review)
→ cosine similarity
→ ranking
→ top results
```

---

## Добавленные фильтры

Были добавлены:

- min_score
- sentiment filter
- topic filter

---

## Hybrid search

В результате получился hybrid retrieval pipeline:

```text
semantic search
+ sentiment analysis
+ topic filtering
```

---

## Примеры запросов

```http
/reviews/search?q=трансфер
```

```http
/reviews/search?q=плохой трансфер&sentiment=negative
```

```http
/reviews/search?q=вкусная еда&topic=food
```

---

# Проблемы и ограничения

Во время разработки были обнаружены ограничения.

---

## 1. Embeddings не понимают всё идеально

Некоторые semantic связи определяются плохо:

```text
трансфер
↔
автобус
```

---

## 2. Threshold сильно влияет на retrieval

Слишком высокий threshold:
- убирает полезные результаты

Слишком низкий:
- добавляет шум

---

## 3. Topic filtering ограничен top-N topics

Сейчас topic filtering работает по:
- threshold
- MAX_TOPICS

Из-за этого некоторые темы могут не попасть в итоговый response.

---

# Engineering вывод

Embeddings:
- являются фундаментом modern AI retrieval systems
- лежат в основе:
  - RAG
  - vector databases
  - AI search
  - recommendation systems

Но embeddings:
- не являются "магией"
- требуют evaluation
- могут ошибаться
- чувствительны к phrasing
- требуют domain testing

---

# Что было изучено

В рамках недели были изучены:

- SentenceTransformer
- embeddings
- vector representation
- cosine similarity
- semantic search
- retrieval pipeline
- hybrid search
- threshold tuning
- edge cases
- slang handling
- multilingual embeddings

---

# Что дальше

Следующий этап:

```text
mini-RAG
```

Планируется реализовать:

- /reviews/ask
- retrieval + context assembly
- answer generation
- foundation RAG pipeline
