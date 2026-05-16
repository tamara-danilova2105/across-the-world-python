# Choosing Embedding Models

## Почему для embeddings используется именно эта модель

```python
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Эта модель хорошо подходит для semantic search и обучения embeddings, потому что она:

- multilingual
- обучена для semantic similarity
- lightweight
- хорошо работает на CPU
- подходит для retrieval и RAG-задач

---

# Разбор названия модели

## sentence-transformers

Это семейство моделей, специально предназначенных для embeddings.

Такие модели обучаются не на:
- sentiment analysis
- classification
- generation

а именно на задачах:

- semantic similarity
- retrieval
- paraphrase detection
- semantic search

Главная идея:

```text
похожие тексты
→ embeddings близко

непохожие тексты
→ embeddings далеко
```

---

## paraphrase

Модель обучалась понимать перефразировки.

Например:

```text
"Отель был ужасный"
```

и:

```text
"Номер оказался очень плохим"
```

должны получить близкие embeddings.

Это особенно важно для semantic search.

---

## multilingual

Модель поддерживает несколько языков.

Это важно для:
- русских отзывов
- английских отзывов
- mixed-language текста

Без multilingual-модели embeddings могут плохо работать на русском языке.

---

## MiniLM

MiniLM — компактная transformer architecture.

Преимущества:
- быстрее inference
- меньше памяти
- можно запускать локально на CPU
- удобно для обучения и pet-projects

---

## L12

Означает:

```text
12 transformer layers
```

Больше layers:
- обычно лучше качество
- но выше потребление памяти и медленнее inference

---

# Почему эта модель подходит для проекта с отзывами

Проект использует:
- semantic similarity
- поиск похожих отзывов
- embeddings-based retrieval
- foundation для будущего RAG

Модель хорошо подходит для:
- туристических отзывов
- multilingual semantic search
- небольших production-like систем
- обучения embeddings

---

# Как выбирать embedding model

## Шаг 1. Определить задачу

Сначала выбирается не модель, а задача.

---

## Если нужна classification-задача

Например:
- sentiment analysis
- toxicity detection
- topic classification

Тогда используются:

```python
AutoModelForSequenceClassification
```

---

## Если нужны embeddings / semantic search / RAG

Тогда используются:

```text
sentence-transformers models
```

или retrieval-oriented embedding models.

---

## Если нужна генерация текста

Тогда используются:
- LLM
- causal language models

---

# На что смотреть при выборе embedding model

## 1. Multilingual support

Если данные:
- русские
- multilingual
- mixed-language

то multilingual support обязателен.

---

## 2. Semantic similarity training

Модель должна быть обучена на:
- similarity
- retrieval
- paraphrase tasks

Иначе embeddings могут работать плохо.

---

## 3. Размер модели

Small / medium models:
- быстрее
- дешевле
- проще для local inference

Large models:
- лучше качество
- дороже inference

---

## 4. Retrieval benchmarks

Полезно смотреть:
- MTEB leaderboard
- retrieval benchmarks
- semantic search benchmarks

---

## 5. Поддержка sentence embeddings

Некоторые transformer models:
- не оптимизированы под embeddings
- не подходят для semantic search "из коробки"

Например:

```text
bert-base-uncased
```

не является хорошей embedding model без дополнительного fine-tuning.

---

# Популярные embedding models

## Для обучения и pet-projects

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## Более сильные multilingual embeddings

```text
intfloat/multilingual-e5-large
```

---

## Популярные production embeddings

```text
BAAI/bge-m3
```

---

## OpenAI embeddings

```text
text-embedding-3-small
text-embedding-3-large
```

---

# Главный engineering вывод

Embedding model выбирается не по популярности, а под retrieval-задачу.

Нужно учитывать:

- язык данных
- semantic similarity quality
- retrieval quality
- размер модели
- скорость inference
- production constraints

---

# Почему embeddings важны

Embeddings являются фундаментом для:

- semantic search
- RAG
- vector databases
- recommendation systems
- clustering
- duplicate detection
- semantic retrieval

Современные AI search systems строятся именно вокруг embeddings.
