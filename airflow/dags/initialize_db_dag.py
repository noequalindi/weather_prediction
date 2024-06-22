from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, TIMESTAMP
from datetime import datetime, timedelta
from sqlalchemy.sql import func
import logging
import pandas as pd
from airflow import settings
from airflow.models import Connection
from airflow.settings import Session

datasets_path = '/opt/airflow/datasets/'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 21),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'create_and_load_tables_postgres',
    default_args=default_args,
    description='Create tables and load data in PostgreSQL',
    schedule_interval=timedelta(hours=3),
)

def initialize_db():
    logging.info("Iniciando la creación de tablas en la base de datos PostgreSQL.")

    try:
        # Conexión a la base de datos PostgreSQL
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres:5432/airflow')
        metadata = MetaData(bind=engine)

        # Definición de las tablas
        weather_data = Table(
            'weather_data', metadata,
            Column('Date', Date),
            Column('Location', String),
            Column('MinTemp', Float),
            Column('MaxTemp', Float),
            Column('Rainfall', Float),
            Column('Evaporation', Float),
            Column('Sunshine', Float),
            Column('WindGustDir', String),
            Column('WindGustSpeed', Float),
            Column('WindDir9am', String),
            Column('WindDir3pm', String),
            Column('WindSpeed9am', Float),
            Column('WindSpeed3pm', Float),
            Column('Humidity9am', Float),
            Column('Humidity3pm', Float),
            Column('Pressure9am', Float),
            Column('Pressure3pm', Float),
            Column('Cloud9am', Float),
            Column('Cloud3pm', Float),
            Column('Temp9am', Float),
            Column('Temp3pm', Float),
            Column('RainToday', String),
            Column('RainTomorrow', String),
            Column('Year', Integer),
            Column('Month', Integer),
            Column('Season', String),
        )

        rf_metrics = Table(
            'rf_metrics', metadata,
            Column('created_at', TIMESTAMP, server_default=func.now()),
            Column('id', Integer, primary_key=True),
            Column('accuracy_train', Float),
            Column('accuracy_test', Float),
            Column('confusion_matrix', String),
            Column('f1_score', Float),
            Column('roc_auc', Float),
            Column('fpr', String), 
            Column('tpr', String)  
)

        metadata.create_all()

        logging.info("Tablas creadas exitosamente en la base de datos.")
    except Exception as e:
        logging.error(f"Error al crear las tablas en la base de datos: {e}")
        raise

def load_csv_to_db():
    logger = logging.getLogger("airflow.task")
    logger.info("Iniciando la carga de datos del CSV a la base de datos.")

    try:
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres:5432/airflow')
        logger.info("Conexión a la base de datos establecida.")

        df = pd.read_csv(datasets_path + 'viz_data.csv')
        logger.info("Archivo CSV leído exitosamente.")

        df.to_sql('weather_data', engine, if_exists='append', index=False)
        logger.info("Datos del CSV cargados exitosamente en la base de datos.")
    except Exception as e:
        logger.error(f"Error al cargar los datos del CSV a la base de datos: {e}")
        raise

def init_minio():
    session = Session()
    # Verificar si la conexión ya existe
    conn_id = 'minio_connection'
    existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()

    if not existing_conn:
        # Crear una nueva conexión si no existe
        new_conn = Connection(
            conn_id=conn_id,
            conn_type='HTTP',
            host='localhost',
            login='minioadmin',
            password='minioadmin',
            port=9000,
            extra='{"secure": "false"}'
        )
        session.add(new_conn)
        session.commit()
    else:
        print(f"La conexión '{conn_id}' ya existe.")

create_tables_task = PythonOperator(
    task_id='create_tables_task',
    python_callable=initialize_db,
    dag=dag,
)

init_minio_task = PythonOperator(
    task_id = 'init_minio_task',
    python_callable=init_minio
)

load_csv_task = PythonOperator(
    task_id='load_csv_task',
    python_callable=load_csv_to_db,
    dag=dag,
)

create_tables_task >> init_minio_task >> load_csv_task 
