# backend/Dockerfile
FROM continuumio/miniconda3

RUN conda install -y -c conda-forge \
    sqlalchemy \
    python-dotenv \
    onnxruntime \
    fastapi \
    uvicorn \
    pandas \
    python-multipart \
    psycopg2-binary

WORKDIR /app 

COPY ./models /models
COPY ./app /app 
COPY .env /app/.env 

ENV RUNNING_IN_DOCKER=True

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

