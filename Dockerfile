FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

EXPOSE 10000
CMD ["python", "-m", "haggis.web", "--host", "0.0.0.0", "--port", "10000"]
