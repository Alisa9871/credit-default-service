"""
Скрипт обучения двух версий модели для A/B-тестирования.
v1: LogisticRegression (базовая, простая и интерпретируемая)
v2: RandomForestClassifier (более сложная, обычно даёт лучший F1)

Запуск: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import joblib
import os

# Воспроизводимость
RANDOM_STATE = 42

# ── 1. Загрузка данных ──────────────────────────────────────────────────────
# Датасет: Default of Credit Card Clients Dataset (UCI)
# Скачать вручную с https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# и положить в папку data/ как credit_default.csv
# (или запустить notebooks/download_data.ipynb)

DATA_PATH = os.path.join("data", "credit_default.csv")

print("Загружаем датасет...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Файл {DATA_PATH} не найден.")
    print("Генерируем синтетический датасет для демонстрации...")
    # Синтетический датасет с теми же признаками что в UCI
    np.random.seed(RANDOM_STATE)
    n = 3000
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(10000, 800000, n),
        "SEX": np.random.randint(1, 3, n),
        "EDUCATION": np.random.randint(1, 5, n),
        "MARRIAGE": np.random.randint(1, 4, n),
        "AGE": np.random.randint(21, 65, n),
        "PAY_0": np.random.randint(-1, 5, n),
        "PAY_2": np.random.randint(-1, 5, n),
        "PAY_3": np.random.randint(-1, 5, n),
        "PAY_4": np.random.randint(-1, 5, n),
        "PAY_5": np.random.randint(-1, 5, n),
        "PAY_6": np.random.randint(-1, 5, n),
        "BILL_AMT1": np.random.randint(0, 200000, n),
        "BILL_AMT2": np.random.randint(0, 200000, n),
        "BILL_AMT3": np.random.randint(0, 200000, n),
        "BILL_AMT4": np.random.randint(0, 200000, n),
        "BILL_AMT5": np.random.randint(0, 200000, n),
        "BILL_AMT6": np.random.randint(0, 200000, n),
        "PAY_AMT1": np.random.randint(0, 50000, n),
        "PAY_AMT2": np.random.randint(0, 50000, n),
        "PAY_AMT3": np.random.randint(0, 50000, n),
        "PAY_AMT4": np.random.randint(0, 50000, n),
        "PAY_AMT5": np.random.randint(0, 50000, n),
        "PAY_AMT6": np.random.randint(0, 50000, n),
        "default.payment.next.month": np.random.binomial(1, 0.22, n)
    })

# ── 2. Подготовка данных ────────────────────────────────────────────────────
# Если датасет UCI: первая строка может быть лишней (заголовок из Excel)
# Убираем строку с индексом ID если она есть
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

TARGET = "default.payment.next.month"

# Признаки для модели
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES].values
y = df[TARGET].values

print(f"Размер датасета: {X.shape}")
print(f"Доля дефолтов: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Нормализация — важна для логистической регрессии
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Обучение v1: Logistic Regression ────────────────────────────────────
print("\n--- Обучаем v1: LogisticRegression ---")
model_v1 = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # датасет немного несбалансирован
    random_state=RANDOM_STATE
)
model_v1.fit(X_train_scaled, y_train)

y_pred_v1 = model_v1.predict(X_test_scaled)
print(classification_report(y_test, y_pred_v1))
print(f"F1 (v1): {f1_score(y_test, y_pred_v1):.4f}")
print(f"Precision (v1): {precision_score(y_test, y_pred_v1):.4f}")
print(f"Recall (v1): {recall_score(y_test, y_pred_v1):.4f}")

# ── 4. Обучение v2: Random Forest ──────────────────────────────────────────
print("\n--- Обучаем v2: RandomForestClassifier ---")
model_v2 = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model_v2.fit(X_train, y_train)   # RF не требует нормализации

y_pred_v2 = model_v2.predict(X_test)
print(classification_report(y_test, y_pred_v2))
print(f"F1 (v2): {f1_score(y_test, y_pred_v2):.4f}")
print(f"Precision (v2): {precision_score(y_test, y_pred_v2):.4f}")
print(f"Recall (v2): {recall_score(y_test, y_pred_v2):.4f}")

# ── 5. Сохранение ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# Сохраняем модели вместе со скейлером и списком признаков
# (чтобы на инференсе применять те же преобразования)
joblib.dump({
    "model": model_v1,
    "scaler": scaler,
    "features": FEATURES,
    "version": "v1",
    "description": "LogisticRegression, class_weight=balanced"
}, "models/model_v1.joblib")

joblib.dump({
    "model": model_v2,
    "scaler": None,       # RF не нуждается в нормализации
    "features": FEATURES,
    "version": "v2",
    "description": "RandomForest, n_estimators=100, max_depth=8"
}, "models/model_v2.joblib")

print("\nМодели сохранены:")
print("  models/model_v1.joblib  (LogisticRegression)")
print("  models/model_v2.joblib  (RandomForest)")
print("\nГотово!")
