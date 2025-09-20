FROM python:alpine

WORKDIR /app

RUN pip install --no-cache-dir flask requests curl_cffi werkzeug loguru python-dotenv gunicorn

COPY . .

ENV PORT=5200
ENV PYTHONUNBUFFERED=1

EXPOSE 5200

CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5200} --workers ${WORKERS:-1} --threads ${THREADS:-8} --worker-class gthread --timeout ${TIMEOUT:-180}"]