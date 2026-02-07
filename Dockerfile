FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Insta360 Media SDK — place manually in vendor/insta360/ before build
COPY vendor/insta360/ /opt/insta360/
ENV PATH="/opt/insta360:${PATH}"

RUN mkdir -p /data/raw /data/work /data/output

ENTRYPOINT ["wpv"]
