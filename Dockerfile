# Базовый образ: официальный Python slim (меньше размер, чем полный образ)
FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Сначала копируем только requirements.txt, чтобы использовать кэш Docker
# (если код изменился, но зависимости те же — слой с pip install не пересобирается)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модели
COPY app/ ./app/
COPY models/ ./models/

# Порт, на котором работает сервис
EXPOSE 5000

# Запускаем через gunicorn — он умеет запускать несколько воркеров,
# что решает проблему GIL (как объяснялось на лекции про uWSGI).
# Для учебного проекта 2 воркера достаточно.
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app.api:app"]
