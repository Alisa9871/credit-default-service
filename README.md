# Сервис прогнозирования дефолта по кредитным картам

Итоговый проект по дисциплине «Внедрение моделей машинного обучения»
(Skillfactory, МИФИ, 2026)

## О проекте

Банк хочет заранее знать, какие клиенты с высокой вероятностью не выплатят
долг по кредитной карте в следующем месяце. Это позволяет либо заранее
предупредить клиента, либо ограничить лимит — в общем, снизить потери.

Я реализовала ML-сервис, который принимает данные о клиенте и возвращает
прогноз: дефолт или нет, и с какой вероятностью.

**Датасет:** [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
— данные о 30 000 клиентов кредитных карт на Тайване.

**Две версии модели (для A/B-теста):**
- `v1` — LogisticRegression (простая, быстрая, интерпретируемая)
- `v2` — RandomForestClassifier (сложнее, потенциально лучше качество)

---

## Структура репозитория

```
.
├── app/
│   └── api.py              # Flask-сервис (эндпоинты /predict и /health)
├── models/
│   ├── model_v1.joblib     # обученная модель v1 (LogisticRegression)
│   └── model_v2.joblib     # обученная модель v2 (RandomForest)
├── data/
│   └── credit_default.csv  # датасет (скачать отдельно, см. ниже)
├── configs/
│   └── nginx.conf          # конфиг nginx для docker-compose
├── tests/
│   └── test_api.py         # базовые тесты эндпоинтов
├── train_model.py          # скрипт обучения моделей
├── requirements.txt
├── Dockerfile
├── docker-compose.yml      # опционально: запуск с nginx
├── ARCHITECTURE.md         # архитектурные решения и MLOps-концепты
└── AB_TEST_PLAN.md         # план A/B-тестирования
```

---

## Быстрый старт (локально)

### 1. Клонируем репозиторий

```bash
git clone https://github.com/<your-username>/credit-default-service.git
cd credit-default-service
```

### 2. Создаём виртуальное окружение и устанавливаем зависимости

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Обучаем модели

Если есть датасет — положите файл в `data/credit_default.csv`.
Если нет — скрипт сам сгенерирует синтетические данные для демонстрации.

```bash
python train_model.py
```

После этого в папке `models/` появятся `model_v1.joblib` и `model_v2.joblib`.

### 4. Запускаем сервис

```bash
python app/api.py
```

Сервис доступен на `http://localhost:5000`.

---

## Запуск в Docker

### Собрать и запустить контейнер

```bash
docker build -t credit-default-service .
docker run -p 5000:5000 credit-default-service
```

### Или использовать готовый образ с Docker Hub

```bash
docker pull <dockerhub-username>/credit-default-service:latest
docker run -p 5000:5000 <dockerhub-username>/credit-default-service:latest
```

### С nginx (docker-compose, бонус)

```bash
docker-compose up --build
```

Сервис будет доступен на `http://localhost` (порт 80 через nginx).

---

## API

### GET /health

Проверка работоспособности сервиса.

```bash
curl http://localhost:5000/health
```

Ответ:
```json
{
  "status": "ok",
  "models_loaded": ["v1", "v2"],
  "default_version": "v1",
  "timestamp": "2026-06-12T14:00:00Z"
}
```

---

### POST /predict

Прогнозирование дефолта. По умолчанию использует модель `v1`.
Для A/B-теста можно указать `?version=v2`.

**Формат запроса:**
```json
{
  "features": [<23 числовых признака>]
}
```

**Порядок признаков (важен!):**

| № | Признак | Описание |
|---|---------|----------|
| 1 | LIMIT_BAL | Кредитный лимит (TWD) |
| 2 | SEX | Пол (1=мужчина, 2=женщина) |
| 3 | EDUCATION | Образование (1=аспирантура, 2=университет, 3=школа, 4=другое) |
| 4 | MARRIAGE | Семейное положение (1=женат/замужем, 2=холост, 3=другое) |
| 5 | AGE | Возраст |
| 6–11 | PAY_0..PAY_6 | Статус выплат за последние 6 мес. (-1=вовремя, 1–9=задержка) |
| 12–17 | BILL_AMT1..6 | Суммы по счетам за 6 мес. |
| 18–23 | PAY_AMT1..6 | Суммы выплат за 6 мес. |

**Пример запроса (модель v1):**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "features": [500000, 1, 2, 1, 35,
                    -1, -1, -1, -1, -1, -1,
                    100000, 90000, 80000, 70000, 60000, 50000,
                    5000, 5000, 5000, 5000, 5000, 5000]
     }'
```

**Пример ответа:**
```json
{
  "prediction": 0,
  "probability": 0.2341,
  "model_version": "v1",
  "interpretation": "дефолт маловероятен"
}
```

**Пример запроса (модель v2 для A/B-теста):**
```bash
curl -X POST "http://localhost:5000/predict?version=v2" \
     -H "Content-Type: application/json" \
     -d '{"features": [500000,1,2,1,35,-1,-1,-1,-1,-1,-1,100000,90000,80000,70000,60000,50000,5000,5000,5000,5000,5000,5000]}'
```

---

## Метрики моделей

*(на синтетическом датасете, реальные метрики будут выше)*

| Метрика | v1 (LR) | v2 (RF) |
|---------|---------|---------|
| F1 (дефолт) | 0.315 | 0.093 |
| Precision | 0.230 | 0.186 |
| Recall | 0.504 | 0.062 |

На реальном UCI-датасете RandomForest обычно показывает F1 около 0.50–0.55.
На синтетических данных признаки случайные, поэтому метрики низкие — это ожидаемо.

---

## Документация по архитектуре

Подробнее о выборе архитектуры, концептах MLOps и плане A/B-теста:
- [ARCHITECTURE.md](./ARCHITECTURE.md) — монолит vs микросервисы, uWSGI + NGINX, RabbitMQ, DVC, MLflow
- [AB_TEST_PLAN.md](./AB_TEST_PLAN.md) — постановка теста, метрики, статистический анализ
