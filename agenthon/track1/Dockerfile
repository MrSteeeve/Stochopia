# Track 1 agent image. Build from agenthon/track1:
#   docker build --platform linux/amd64 -t t1agent:dev .
# Scoring invocation: <image> solve --task-dir /input --out /app/output
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C library, as in the organizers' finance-bench-sandbox image.
RUN cd /tmp \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr && make -j"$(nproc)" && make install \
    && cd / && rm -rf /tmp/ta-lib*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY t1agent /opt/t1agent/t1agent
ENV PYTHONPATH=/opt/t1agent PYTHONUNBUFFERED=1 MPLBACKEND=Agg
RUN mkdir -p /app/output /app/data /output
WORKDIR /app
ENTRYPOINT ["python", "-m", "t1agent"]
