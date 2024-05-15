FROM python:3.11-slim as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

COPY ./requirements.txt ./requirements.txt

RUN apt -y update && apt install -y gcc

RUN pip3 install --user -r requirements.txt

FROM python:3.11-slim as final

RUN apt -y update && apt install curl -y && apt autoremove -y && apt clean -y && rm -rf /var/lib/apt/lists/*

# copy only the dependencies installation from the 1st stage image
COPY --from=builder /root/.local /root/.local

WORKDIR /app

COPY . .

ENV PATH=/root/.local/bin:$PATH

RUN python ntlk_download.py

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]