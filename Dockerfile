FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true \
 && pip install --no-cache-dir fastapi uvicorn pyyaml jsonschema joblib pandas

COPY . .
ENV CONFIG_PATH=config.yaml
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host","0.0.0.0", "--port","8000"]