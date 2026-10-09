FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 先装依赖再拷源码, 充分利用层缓存
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "reflex_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
