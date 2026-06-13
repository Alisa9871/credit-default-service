"""
Базовые тесты для проверки работоспособности API.
Запуск: python -m pytest tests/test_api.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.api import app

SAMPLE_FEATURES = [
    500000, 1, 2, 1, 35,
    -1, -1, -1, -1, -1, -1,
    100000, 90000, 80000, 70000, 60000, 50000,
    5000, 5000, 5000, 5000, 5000, 5000
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_ok(client):
    """Эндпоинт /health должен возвращать статус ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "v1" in data["models_loaded"]
    assert "v2" in data["models_loaded"]


def test_predict_v1(client):
    """Предсказание моделью v1 должно вернуть 0 или 1."""
    r = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200
    data = r.get_json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "v1"


def test_predict_v2(client):
    """Предсказание моделью v2 должно работать через query-параметр."""
    r = client.post("/predict?version=v2", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 200
    data = r.get_json()
    assert data["model_version"] == "v2"


def test_predict_wrong_version(client):
    """Несуществующая версия модели должна вернуть 400."""
    r = client.post("/predict?version=v99", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 400


def test_predict_missing_features(client):
    """Запрос без поля features должен вернуть 400."""
    r = client.post("/predict", json={"data": SAMPLE_FEATURES})
    assert r.status_code == 400


def test_predict_wrong_feature_count(client):
    """Неправильное количество признаков должно вернуть 400."""
    r = client.post("/predict", json={"features": [1, 2, 3]})
    assert r.status_code == 400
