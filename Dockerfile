FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

RUN echo "Downloading model for Run ID: ${RUN_ID}" > /tmp/model_download.log

CMD ["sh", "-c", "echo Starting container with model Run ID: ${RUN_ID} && cat /tmp/model_download.log"]