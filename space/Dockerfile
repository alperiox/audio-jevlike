FROM python:3.11-slim

# ffmpeg decodes the browser's webm/opus recording to 16 kHz mono PCM.
# Without it every request fails at to_wav16k() with FileNotFoundError.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000; HF_HOME must be writable or the first
# from_pretrained() call dies on a read-only cache dir.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
EXPOSE 7860

CMD ["python", "app.py", "--ckpt", "model/best.pt", \
     "--bins", "model/bins.json", "--page", "live_demo.html"]
