# Build from the project root:  docker build -f app/Dockerfile -t nmt-en-am .
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentencepiece fastapi uvicorn pydantic pandas pyyaml
COPY src ./src
COPY app ./app
COPY checkpoints ./checkpoints
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
