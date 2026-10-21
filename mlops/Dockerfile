FROM python:3.14-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY mlops/requirements-api.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mlops/api.py mlops/predict.py /app/

ARG RUN_ID
COPY mlops/artifacts/runs/${RUN_ID}/ /app/model/

ENV MODEL_PATH=/app/model/model.joblib
ENV MODEL_RUN_ID=${RUN_ID}

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
