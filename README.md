Сервис и CLI для **кластеризации и авто-категоризации документов** (KMeans по эмбеддингам Sentence-Transformers), 
получения **топ-термов** для каждого кластера (CT-TFIDF/YAKE/KeyBERT), а также **предсказания распределения вероятностей по кластерам** и 
**zero-shot типа документа**. REST-API на FastAPI. :contentReference[oaicite:0]{index=0}

API поддерживает загрузку **отдельных файлов**, **ZIP-архива**, а также работу по относительным путям в пределах базовой директории 
(контролируется `DOCS_BASE`). Предусмотрены эндпоинты: `/fit`, `/predict`, `/classify`, `/doc_type`, `/health`. :contentReference[oaicite:1]{index=1}

---

## Основные возможности

- **FIT (обучение/кластеризация)**: автоматический выбор *k* (силуэт), опциональный рекурсивный **recluster** крупных кластеров, 
  пост-слияние мелких кластеров, генерация *categories.json*, *assignments.jsonl* и модели центроидов (*semantic_centroids.json*). :contentReference[oaicite:2]{index=2}  
- **PREDICT**: для новых документов — метка кластера, **sim_to_centroid**, **scores** (softmax по косинусам), ключевые теги (YAKE/KeyBERT), 
  **zero-shot тип документа** (по списку типов). :contentReference[oaicite:3]{index=3}
- **REST-API**: эндпоинты `/fit`, `/predict`, `/classify` (прямое отнесение к переданным меткам: sim или NLI), `/doc_type` (готовый набор типов), `/health`. 
  Поддержка загрузки `files[]`/`bundle` и чтения из `input_dir`/`paths`. Базовая директория ограничена переменной окружения `DOCS_BASE`. :contentReference[oaicite:4]{index=4}

---

## Структура

- `auto_categorize.py` — CLI с режимами **fit/predict**; внутри чтение/очистка текста, эмбеддинги, KMeans, рекурсивный recluster, 
  CT-TFIDF/YAKE/KeyBERT для лейблов и тегов, zero-shot типов документов, сохранение модели центроидов. :contentReference[oaicite:5]{index=5}  
- `app.py` — FastAPI-сервис: сбор входа (files/bundle/paths/input_dir), запуск `auto_categorize.py` как подпроцесса, 
  типовые ответы и модели pydantic; есть служебные `/health`, универсальный `/classify` (sim/NLI). Путь ограничивается `DOCS_BASE` (по умолчанию `/data`). :contentReference[oaicite:6]{index=6}

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Рекомендуемые зависимости (requirements.txt):

scss
Копировать код
fastapi
uvicorn[standard]
numpy
scikit-learn
sentence-transformers
transformers
pdfminer.six
python-multipart
yake           # опционально (теги)
keybert        # опционально (теги)
pymorphy2      # опционально (лемматизация RU)
Для sentence-transformers может понадобиться torch (CPU/GPU — установите подходящий билд).

Запуск API
bash
Копировать код
export DOCS_BASE=$PWD/data      # база, внутри которой можно читать/писать
mkdir -p "$DOCS_BASE"
uvicorn app:app --reload --port 8000
Проверка: http://localhost:8000/docs (Swagger). app

Примеры запросов (cURL)
1) Обучение (кластеризация) — /fit
Загрузка одного файла:

bash
Копировать код
curl -s -X POST "http://localhost:8000/fit" \
  -F "files=@Docs_test/1.txt" \
  -F "out_root=runs" \
  -F "device=cpu" | jq .
Загрузка ZIP:

bash
Копировать код
curl -s -X POST "http://localhost:8000/fit" \
  -F "bundle=@Docs_test.zip" \
  -F "out_root=runs" \
  -F "predict_on_fit=true" \
  -F "prob_temperature=0.8" | jq .
Работа по каталогу внутри DOCS_BASE:

bash
Копировать код
curl -s -X POST "http://localhost:8000/fit" \
  -F "input_dir=Docs_test" \
  -F "out_root=runs" \
  -F "k=7" | jq .
Ответ содержит run_dir, categories, assignments, predictions (если включён predict_on_fit) и список doc_types. app

2) Предсказание на новых документах — /predict
С моделью из run_dir/models/semantic_centroids.json:

bash
Копировать код
curl -s -X POST "http://localhost:8000/predict" \
  -F "input_dir=Docs_new" \
  -F "model_path=runs/<RUN_ID>/models/semantic_centroids.json" \
  -F "device=cpu" \
  -F "prob_temperature=1.0" | jq .
Можно передать модель файлом: -F "model=@semantic_centroids.json". В predictions.jsonl будут поля pred_label, sim_to_centroid, scores, tags, а также doc_type_pred/doc_type_scores если задан doc_types. auto_categorize

3) Прямая классификация по меткам — /classify
SIM-бэкенд (эмбеддинги):

bash
Копировать код
curl -s -X POST "http://localhost:8000/classify" \
  -F "input_dir=Docs_test" \
  -F 'labels_json=["Счёт","Договор","Заявление"]' \
  -F "backend=sim" | jq .
NLI-бэкенд (zero-shot):

bash
Копировать код
curl -s -X POST "http://localhost:8000/classify" \
  -F "input_dir=Docs_test" \
  -F 'labels_json=["Счёт","Договор","Заявление"]' \
  -F "backend=nli" \
  -F "nli_model=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" | jq .
Каждый элемент содержит scores по меткам и pred_label/pred_prob. app

4) Готовые типы документов — /doc_type
bash
Копировать код
curl -s -X POST "http://localhost:8000/doc_type" \
  -F "input_dir=Docs_test" | jq .
Если не передавать список типов, используется дефолтный набор на RU. app

CLI без API (напрямую)
Обучение (fit)
bash
Копировать код
python auto_categorize.py fit Docs_test \
  --device cpu \
  --min-k 7 --max-k 12 --k-select-tol 0.25 \
  --recluster-large --recluster-max-depth 1 \
  --recluster-min-size 6 --recluster-child-min-size 2 \
  --recluster-min-k 2 --recluster-max-k 4 --recluster-tol 0.05 --recluster-max-dominance 0.90 \
  --df-stop-threshold 0.6 --label-local-stop-topk 2 --label-use-yake \
  --tags-backend yake \
  --out-cats categories.json \
  --out-assign assignments.jsonl \
  --save-model models/semantic_centroids.json \
  --predict-on-fit --predict-out predictions.jsonl \
  --prob-temperature 0.8 --doc-types "" --doc-types-temperature 0.7 --doc-types-out doc_types.txt
auto_categorize

Предсказание (predict)
bash
Копировать код
python auto_categorize.py predict Docs_new \
  --model models/semantic_centroids.json \
  --device cpu \
  --prob-temperature 1.0 \
  --doc-types "" --doc-types-temperature 0.7 \
  --out predictions.jsonl
auto_categorize

Переменные окружения
DOCS_BASE — база разрешённых путей для API; все input_dir/paths нормализуются внутрь неё. По умолчанию /data.
Это удобно в Docker (маппим -v /host/data:/data). app

Docker (пример)
dockerfile
Копировать код
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DOCS_BASE=/data
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
Запуск:

bash
Копировать код
docker build -t docsvc .
docker run --rm -p 8000:8000 -e DOCS_BASE=/data -v $PWD/data:/data docsvc
Лицензия
MIT / Internal 
