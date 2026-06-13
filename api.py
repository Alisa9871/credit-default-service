"""
Веб-сервис для прогнозирования дефолта по кредитным картам.

Эндпоинты:
  GET  /health          — проверка работоспособности
  POST /predict         — предсказание (использует модель по умолчанию v1)
  POST /predict?version=v2  — предсказание моделью v2 (для A/B-тест)

Автор: студент магистратуры
Курс: Внедрение моделей ML, Skillfactory
"""

import os
import json
import logging
import datetime

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Логирование ──────────────────────────────────────────────────────────────
# Настраиваем JSON-логирование: в production такие логи удобно собирать
# через ELK-стек (Elasticsearch + Logstash + Kibana)
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# ── Загрузка моделей ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model(version: str):
    """Загрузить модель нужной версии из .joblib файла."""
    path = os.path.join(MODELS_DIR, f"model_{version}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    artifact = joblib.load(path)
    logger.info(f"Загружена модель {version}: {artifact['description']}")
    return artifact

# Загружаем обе модели при старте сервиса, чтобы не делать это при каждом запросе
try:
    MODEL_REGISTRY = {
        "v1": load_model("v1"),
        "v2": load_model("v2"),
    }
    DEFAULT_VERSION = "v1"
    logger.info("Обе модели загружены успешно.")
except FileNotFoundError as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    MODEL_REGISTRY = {}
    DEFAULT_VERSION = None


# ── Вспомогательные функции ──────────────────────────────────────────────────
def preprocess(data: list, artifact: dict) -> np.ndarray:
    """Применить те же преобразования, что при обучении."""
    X = np.array(data, dtype=float).reshape(1, -1)
    if artifact["scaler"] is not None:
        X = artifact["scaler"].transform(X)
    return X


# ── Эндпоинты ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Проверка работоспособности сервиса.
    Возвращает статус и список доступных версий моделей.
    """
    status = "ok" if MODEL_REGISTRY else "error"
    return jsonify({
        "status": status,
        "models_loaded": list(MODEL_REGISTRY.keys()),
        "default_version": DEFAULT_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }), 200 if status == "ok" else 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Прогнозирование дефолта по кредитной карте.

    Query-параметр:
      version (str): "v1" или "v2", по умолчанию "v1"

    Тело запроса (JSON):
      {
        "features": [<23 числовых признака в порядке обучения>]
      }

    Ответ:
      {
        "prediction": 0 или 1,
        "probability": float,  // вероятность дефолта (класс 1)
        "model_version": "v1"
      }

    Признаки (23 штуки, в этом порядке):
      LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
      PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
      BILL_AMT1-6, PAY_AMT1-6
    """
    # Выбираем версию модели (для A/B-тестирования)
    version = request.args.get("version", DEFAULT_VERSION)

    if version not in MODEL_REGISTRY:
        return jsonify({
            "error": f"Версия модели '{version}' не найдена. Доступны: {list(MODEL_REGISTRY.keys())}"
        }), 400

    # Парсим тело запроса
    body = request.get_json(silent=True)
    if body is None or "features" not in body:
        return jsonify({
            "error": "Тело запроса должно содержать JSON с полем 'features'"
        }), 400

    features = body["features"]

    artifact = MODEL_REGISTRY[version]
    expected_n = len(artifact["features"])

    if len(features) != expected_n:
        return jsonify({
            "error": f"Ожидается {expected_n} признаков, получено {len(features)}"
        }), 400

    try:
        X = preprocess(features, artifact)
        model = artifact["model"]

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        result = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": version,
            "interpretation": "дефолт вероятен" if prediction == 1 else "дефолт маловероятен"
        }

        # Логируем каждый запрос в JSON-формате
        logger.info(json.dumps({
            "event": "prediction",
            "version": version,
            "prediction": prediction,
            "probability": round(probability, 4)
        }))

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    # Запуск напрямую (dev-режим)
    # В production используем uWSGI или gunicorn (см. ARCHITECTURE.md)
    app.run(host="0.0.0.0", port=5000, debug=False)
