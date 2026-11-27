FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/graduate_admission.csv data/graduate_admission.csv

# Build the data and the model inside the image so it is reproducible
RUN python src/prepare_data.py && python src/train.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
