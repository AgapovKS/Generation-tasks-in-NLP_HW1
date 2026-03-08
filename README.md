# Generation-tasks-in-NLP_HW1

## Описание проекта

Чат-бот на основе подхода Retrieval-Based: вместо генерации новых текстов бот находит
наиболее семантически близкую реплику из базы знаний— реальных фраз Рика Санчеза из
сериала *Rick and Morty*.

---

## Данные

**Источник:** [`ysharma/rickandmorty`](https://huggingface.co/datasets/ysharma/rickandmorty) (HuggingFace)

**Что использовалось:** только реплики персонажа **Rick** (и его аналоги).

**Структура датасета:** index, season no., episode no., episode name, name, line — скрипты сезонов 1–3.
- Удаление ремарок в скобках: `(burps)`, `[sound]`
- Фильтрация слишком коротких реплик (< 5 слов) и длинных (> 300 символов)
- Дедупликация
- Итоговый датасет: 451 реплика

---

## Архитектура

```
Запрос пользователя
       │
       ▼
 SentenceTransformer
 (paraphrase-multilingual-MiniLM-L12-v2)
       │
       ▼  384-мерный вектор
 FAISS IndexFlatIP
 (косинусное сходство = inner product при L2-нормализации)
       │
       ▼
 Топ-K наиболее похожих реплик + similarity score
       │
       ▼
     Ответ
```

### Модель эмбеддингов

| Параметр | Значение |
|---|---|
| Модель | `paraphrase-multilingual-MiniLM-L12-v2` |
| Размер | ~118 MB |
| Размерность | 384 |
| Языки | 50+, включая русский и английский |
| Нормализация | L2 (cosine sim = dot product) |

### Индексы FAISS

- **IndexFlatIP** — точный поиск, O(N) на запрос. 
- **IndexIVFFlat** — приближённый поиск (ANN), быстрее на больших данных. `nlist=100`, `nprobe=10`.

### Keyword Search Baseline

Реализован через **BM25Okapi** (`rank-bm25`). Классический TF-IDF-подобный поиск по
ключевым словам. Не требует GPU и работает мгновенно, но не понимает смысл.


---

## Бенчмарк скорости

| Метод | Среднее время/запрос |
|---|---|
| FAISS IndexFlatIP (exact) | 1.54 мс |
| FAISS IndexIVFFlat (ANN) | 2.07 мс |
| FAISS + кэш запросов | 0.47 мс |
| BM25 | 0.44 мс |

---

## Выводы о качестве Retrieval-подхода

### Достоинства
- Просто и интерпретируемо: всегда видна мера сходства
- Не галлюцинирует — отвечает реальными цитатами персонажа
- Работает без GPU на CPU

### Ограничения
- Ограничен размером базы: если в базе нет подходящей реплики,
  бот вернёт "ближайшее", которое может не соответствовать контексту
-  Каждый запрос обрабатывается независимо, контекст не учитывается
- Технически близкие по вектору фразы могут
  быть далеки по смыслу в контексте диалога

---

## Инструкция по запуску

### Google Colab

1. Загрузите `HW_NLP_TASK_1_Agapov_KS.ipynb` в Google Colab
2. Запустите ячейку с установкой зависимостей:
   ```python
   !pip install sentence-transformers faiss-cpu datasets pandas numpy scikit-learn rank-bm25 tqdm
   ```
3. Запустите все ячейки по порядку (`Runtime → Run all`)
4. В ячейке 9 раскомментируйте `interactive_chat()` для интерактивного режима

### Локально

```bash
# Клонируем / скачиваем проект
cd rick_chatbot

# Создаём окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Запускаем ноутбук
jupyter notebook rick_chatbot.ipynb
```

### Структура проекта

```
rick_chatbot/
├── rick_chatbot.ipynb   # Основной ноутбук
├── requirements.txt     # Зависимости
├── README.md            # Этот отчёт
├── rick_lines.csv       # Датасет (генерируется при запуске)
├── embeddings.npy       # Матрица эмбеддингов (генерируется)
├── faiss_exact.index    # FAISS exact index (генерируется)
└── faiss_ivf.index      # FAISS IVF index (генерируется)
```
