FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLIP_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tools ./tools
COPY sample_audio ./sample_audio

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 5000

# 工程状态保存在进程内存中，必须单 worker；用线程处理并发，每工程有锁
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:5000", "app.main:app"]
