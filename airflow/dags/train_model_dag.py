from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import joblib
import logging
import json
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError
from airflow.exceptions import AirflowException
from airflow.hooks.base_hook import BaseHook
from io import BytesIO
import boto3
import os


MINIO_CONN_ID = 'minio_connection'
MINIO_BUCKET = 'weather-prediction-s3'
MINIO_MODELS_PREFIX = 'models/'
MINIO_DATASETS_PREFIX = 'datasets/'

X_train_path = 'datasets/X_train.csv'
X_test_path = 'datasets/X_test.csv'
y_train_path = 'datasets/y_train.csv'
y_test_path = 'datasets/y_test.csv'

DATASETS_DIR = '/opt/airflow/datasets'

       
s3_client = boto3.client(
    's3',
    endpoint_url="http://s3:9000",
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=boto3.session.Config(signature_version='s3v4')
)

def upload_csv_to_minio(file_name, minio_bucket, s3_client):
    try:
        file_path = os.path.join(DATASETS_DIR, file_name)
        with open(file_path, 'rb') as data:
            s3_client.put_object(Bucket=minio_bucket, Key=f'datasets/{file_name}', Body=data)
        logging.info(f"Archivo {file_name} subido correctamente a MinIO en el bucket {minio_bucket}")
    except Exception as e:
        logging.error(f"Error al subir archivo {file_name} a MinIO: {e}")
        raise AirflowException(f"Error al subir archivo {file_name} a MinIO: {e}")

def upload_csv_files_to_minio():
    try:
        files_to_upload = os.listdir(DATASETS_DIR)
        for file_name in files_to_upload:
            if file_name.endswith('.csv'):
                upload_csv_to_minio(file_name, MINIO_BUCKET, s3_client)
    except Exception as e:
        logging.error(f"Error en la carga de archivos CSV a MinIO: {e}")
        raise AirflowException(f"Error en la carga de archivos CSV a MinIO: {e}")

def load_data_from_minio(file_path):
    try:
        response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=file_path)
        return pd.read_csv(response['Body'])
    except Exception as e:
        logging.error(f"Error al cargar datos desde MinIO. File path: {file_path}, Bucket: {MINIO_BUCKET}. Detalles: {e}")
        raise  # Re-lanza la excepción para que Airflow maneje el error correctamente

def save_model_to_minio(model, filename):
    try:
        model_buffer = BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        s3_client.put_object(
            Bucket=MINIO_BUCKET,
            Key=f"{MINIO_MODELS_PREFIX}{filename}",
            Body=model_buffer.getvalue()
        )
        logging.info(f"Modelo guardado en MinIO: {MINIO_BUCKET}/{MINIO_MODELS_PREFIX}{filename}")
    except Exception as e:
        logging.error(f"Error al guardar el modelo en MinIO: {e}")
        raise AirflowException(f"Error al guardar el modelo en MinIO: {e}")

def train_random_forest():
    logging.info("Iniciando el entrenamiento del modelo Random Forest.")
    
    try:
        # Cargar los datos de entrenamiento y prueba desde MinIO
        X_train = load_data_from_minio(X_train_path)
        X_test = load_data_from_minio(X_test_path)
        y_train = load_data_from_minio(y_train_path)
        y_test = load_data_from_minio(y_test_path)

        # Entrenar el modelo Random Forest
        best_model_rf = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=2, max_depth=None, criterion='entropy')
        best_model_rf.fit(X_train, y_train.values.ravel())

        # Guardar el modelo entrenado en MinIO
        save_model_to_minio(best_model_rf, 'best_model_rf.pkl')

        # Convertir el modelo a ONNX y guardarlo en MinIO
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(best_model_rf, initial_types=initial_type)
        onnx_model_buffer = BytesIO()
        onnx_model_buffer.write(onnx_model.SerializeToString())
        onnx_model_buffer.seek(0)
        s3_client.put_object(
            Bucket=MINIO_BUCKET,
            Key=f"{MINIO_MODELS_PREFIX}best_random_forest_model.onnx",
            Body=onnx_model_buffer.getvalue()
        )
        logging.info(f"Modelo Random Forest convertido a ONNX y guardado en MinIO: {MINIO_BUCKET}/{MINIO_MODELS_PREFIX}best_random_forest_model.onnx")

        # Predicciones y métricas
        y_pred_rf_best_train = best_model_rf.predict(X_train)
        y_pred_rf_best = best_model_rf.predict(X_test)

        accuracy_rf_best_train = accuracy_score(y_train, y_pred_rf_best_train)
        accuracy_rf_best = accuracy_score(y_test, y_pred_rf_best)
        confusion_matrix_rf_best = confusion_matrix(y_test, y_pred_rf_best)
        f1_rf_best = f1_score(y_test, y_pred_rf_best)
        y_pred_proba_rf_best = best_model_rf.predict_proba(X_test)[:, 1]
        
        # Calcular FPR, TPR y ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf_best)
        roc_auc = roc_auc_score(y_test, y_pred_proba_rf_best)

        # Guardar las métricas en la base de datos
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres:5432/airflow')
        metadata = MetaData()

        rf_metrics = Table(
            'rf_metrics', metadata,
            Column('id', Integer, primary_key=True),
            Column('created_at', DateTime, server_default=func.now()),
            Column('accuracy_train', Float),
            Column('accuracy_test', Float),
            Column('confusion_matrix', String),
            Column('f1_score', Float),
            Column('roc_auc', Float),
            Column('fpr', String),
            Column('tpr', String)
        )

        # Utilizar inspect para verificar la existencia de la tabla
        inspector = inspect(engine)
        if not inspector.has_table('rf_metrics'):
            metadata.create_all(engine)

        with engine.connect() as conn:
            conn.execute(
                rf_metrics.insert().values(
                    accuracy_train=accuracy_rf_best_train,
                    accuracy_test=accuracy_rf_best,
                    confusion_matrix=str(confusion_matrix_rf_best.tolist()),
                    f1_score=f1_rf_best,
                    roc_auc=roc_auc,
                    fpr=json.dumps(fpr.tolist()),  # Guardar como JSON
                    tpr=json.dumps(tpr.tolist())   # Guardar como JSON
                )
            )

        logging.info("Entrenamiento del modelo Random Forest completado y métricas guardadas en la base de datos.")
    except SQLAlchemyError as e:
        logging.error(f"Error al guardar las métricas en la base de datos: {e}")
        raise AirflowException(f"Error al guardar las métricas en la base de datos: {e}")
    except Exception as e:
        logging.error(f"Error en el entrenamiento del modelo Random Forest: {e}")
        raise AirflowException(f"Error en el entrenamiento del modelo Random Forest: {e}")



# Definir el DAG
dag = DAG(
    'train_random_forest_to_minio',
    schedule_interval=None,
    start_date=datetime(2024, 6, 20),
    description='Entrenar modelo Random Forest y guardar en MinIO',
    catchup=False
)

# Definir los operadores
upload_csv_task = PythonOperator(
    task_id='upload_csv_files',
    python_callable=upload_csv_files_to_minio,
    dag=dag,
)

train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    dag=dag,
)

# Definir dependencias
upload_csv_task >> train_rf_task
