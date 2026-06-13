# Сервис прогнозирования дефолта по кредитным картам

Итоговый проект по дисциплине "Внедрение моделей машинного обучения" (Skillfactory, МИФИ, 2026)

Docker Hub: https://hub.docker.com/r/alisa123lp/credit-default-service

GitHub: https://github.com/Alisa9871/credit-default-service

---

## О проекте

Задача: предсказать, какой клиент не заплатит по кредитке в следующем месяце. Банк может использовать это чтобы заранее связаться с клиентом или ограничить лимит.

Я сделала ML-сервис, который принимает данные о клиенте и возвращает прогноз: дефолт или нет, и с какой вероятностью.

Датасет: [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) -- 30 000 клиентов кредитных карт на Тайване.

Две версии модели (для A/B-теста):
- `v1` -- LogisticRegression (простая, быстрая)
- `v2` -- RandomForestClassifier (посложнее, должна быть точнее)

---

## Структура репозитория

```
.
├── app/
│   └── api.py           # Flask-сервис (/predict и /health)
├── models/
│   ├── model_v1.joblib  # модель v1 (LogisticRegression)
│   └── model_v2.joblib  # модель v2 (RandomForest)
├── configs/
│   └── nginx.conf       # конфиг nginx для docker-compose
├── tests/
│   └── test_api.py      # тесты
├── train_model.py       # скрипт обучения
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md      # архитектура и MLOps концепты
└── AB_TEST_PLAN.md      # план A/B-тестирования
```

---

## Как запустить локально

### 1. Клонируем репозиторий

```bash
git clone https://github.com/Alisa9871/credit-default-service.git
cd credit-default-service
```

### 2. Устанавливаем зависимости

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Обучаем модели

```bash
python train_model.py
```

### 4. Запускаем сервис

```bash
python app/api.py
```

Сервис работает на `http://localhost:5000`.

---

## Запуск в Docker

```bash
docker build -t credit-default-service .
docker run -p 5000:5000 credit-default-service
```

Или скачать готовый образ:

```bash
docker pull alisa123lp/credit-default-service
docker run -p 5000:5000 alisa123lp/credit-default-service
```

С nginx (бонус):

```bash
docker-compose up --build
```

---

## API

### GET /health

```bash
curl http://localhost:5000/health
```

Ответ:
```json
{
  "status": "ok",
  "models_loaded": ["v1", "v2"],
  "default_version": "v1",
  "timestamp": "2026-06-13T14:00:00Z"
}
```

### POST /predict

Принимает JSON с 23 признаками клиента, возвращает прогноз.

Пример запроса (модель v1):
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [500000,1,2,1,35,-1,-1,-1,-1,-1,-1,100000,90000,80000,70000,60000,50000,5000,5000,5000,5000,5000,5000]}'
```

Пример запроса (модель v2):
```bash
curl -X POST "http://localhost:5000/predict?version=v2" \
     -H "Content-Type: application/json" \
     -d '{"features": [500000,1,2,1,35,-1,-1,-1,-1,-1,-1,100000,90000,80000,70000,60000,50000,5000,5000,5000,5000,5000,5000]}'
```

Ответ:
```json
{
  "prediction": 0,
  "probability": 0.2341,
  "model_version": "v1",
  "interpretation": "дефолт маловероятен"
}
```

Порядок признаков:

| N | Признак | Описание |
|---|---------|----------|
| 1 | LIMIT_BAL | Кредитный лимит |
| 2 | SEX | Пол (1=мужчина, 2=женщина) |
| 3 | EDUCATION | Образование |
| 4 | MARRIAGE | Семейное положение |
| 5 | AGE | Возраст |
| 6-11 | PAY_0..PAY_6 | Статус выплат за 6 мес. |
| 12-17 | BILL_AMT1..6 | Суммы по счетам за 6 мес. |
| 18-23 | PAY_AMT1..6 | Суммы выплат за 6 мес. |

---

## Метрики моделей

На реальном UCI датасете:

| Метрика | v1 (LR) | v2 (RF) |
|---------|---------|---------|
| F1 | ~0.47 | ~0.52 |
| Precision | ~0.40 | ~0.55 |
| Recall | ~0.57 | ~0.49 |

---

## Документация

- [ARCHITECTURE.md](./ARCHITECTURE.md) -- архитектура, uWSGI/NGINX, RabbitMQ, DVC, MLflow
- [AB_TEST_PLAN.md](./AB_TEST_PLAN.md) -- план A/B-теста, метрики, статистика
