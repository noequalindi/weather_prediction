from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import onnxruntime as ort
import ast
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select, func, desc, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError
from sqlalchemy.ext.declarative import declarative_base
import boto3
import botocore 
from botocore.exceptions import ClientError
from io import BytesIO
import json
import requests
from requests.auth import HTTPBasicAuth

running_in_docker = os.getenv('RUNNING_IN_DOCKER', 'False') == 'True'
load_dotenv()
app = FastAPI()

# Set database connection URL
if running_in_docker:
    DATABASE_URL = 'postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
else:
    DATABASE_URL = 'postgresql+psycopg2://airflow:airflow@localhost:5432/airflow'

print(f"DATABASE_URL is set to: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
metadata = MetaData()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

MINIO_BUCKET = 'weather-prediction-s3'
MINIO_MODELS_PREFIX = 'models/'
MINIO_ENDPOINT_URL = 'http://localhost:9000'

s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT_URL,
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=boto3.session.Config(signature_version='s3v4')
)
s3_bucket = "weather-prediction-s3"

def load_onnx_model_from_minio(model_filename, dag_id):
    s3_key = f"models/{model_filename}"

    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)

        model_bytes = response['Body'].read()
        return model_bytes

    except botocore.exceptions.EndpointConnectionError as e:
        error_message = f"Could not connect to the endpoint URL: {s3_bucket}/{s3_key}"
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo onnx: {error_message}")

    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        
        # Verificar si el error es debido a que el objeto no existe
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"El modelo '{model_filename}' todavía no se encuentra disponible en S3. El DAG '{dag_id}' todavía no se ha ejecutado o no se ha completado con éxito." )
        else:
            raise HTTPException(status_code=500, detail=f"Error al cargar el modelo ONNX: {str(e)}")

class RainPrediction(BaseModel):
    MinTemp: float
    Rainfall: float
    WindSpeed9am: float
    Humidity3pm: float
    RainToday: int

# Configurar CORS para permitir todas las origenes en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

def join_onnx_parts(part0_path, part1_path, part2_path, model_url):
    if running_in_docker:
        output_path = '../models/'+model_url
    else:
        output_path = os.getenv('MODELS_PATH') + model_url
    
    with open(output_path, 'wb') as output_file:
        for part_path in [part0_path, part1_path, part2_path]:
            with open(part_path, 'rb') as part_file:
                output_file.write(part_file.read())

# Rutas de los modelos ONNX
if running_in_docker:
    scaler_model_path = '../models/standard_scaler_model.onnx'
    decision_tree_model_path = '../models/best_decision_tree_model.onnx'
    part0_path = '../models/best_random_forest_model_default.onnx.part0'
    part1_path = '../models/best_random_forest_model_default.onnx.part1'
    part2_path = '../models/best_random_forest_model_default.onnx.part2'
else:
    scaler_model_path = os.getenv('MODELS_PATH') + 'standard_scaler_model.onnx'
    decision_tree_model_path = os.getenv('MODELS_PATH') + 'best_decision_tree_model.onnx'
    part0_path = os.getenv('MODELS_PATH') + 'best_random_forest_model_default.onnx.part0'
    part1_path = os.getenv('MODELS_PATH') + 'best_random_forest_model_default.onnx.part1'
    part2_path = os.getenv('MODELS_PATH') + 'best_random_forest_model_default.onnx.part2'


join_onnx_parts(part0_path, part1_path, part2_path, 'best_random_forest_model_default.onnx')

# Crear sesiones de inferencia para ambos modelos
scaler_session = ort.InferenceSession(scaler_model_path)
decision_tree_session = ort.InferenceSession(decision_tree_model_path)

if running_in_docker:
    output_best_rf_path = '../models/best_random_forest_model_default.onnx' 
else:
    output_best_rf_path = os.getenv('MODELS_PATH') + 'best_random_forest_model_default.onnx'
    
random_forest_default_session = ort.InferenceSession(output_best_rf_path) #modelo RF por default

@app.post("/predict/{model_type}")
async def predict(data: RainPrediction, model_type: str = "decision_tree"):
    # Check if scaler or decision tree sessions are not loaded
    if scaler_session is None or decision_tree_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = np.array([[data.MinTemp, data.Rainfall, data.WindSpeed9am, data.Humidity3pm, data.RainToday]], dtype=np.float32)

    if model_type == "decision_tree":
        # Inference with DecisionTreeClassifier model
        if scaler_session is None:
            raise HTTPException(status_code=500, detail="Scaler model not loaded")

        input_name = scaler_session.get_inputs()[0].name
        output_name = scaler_session.get_outputs()[0].name
        scaler_inputs = {input_name: features}
        scaled_features = scaler_session.run([output_name], scaler_inputs)[0]

        decision_tree_input_name = decision_tree_session.get_inputs()[0].name
        decision_tree_output_name = decision_tree_session.get_outputs()[0].name
        decision_tree_inputs = {decision_tree_input_name: scaled_features}
        prediction = decision_tree_session.run([decision_tree_output_name], decision_tree_inputs)[0]

    elif model_type == "random_forest":
        try:
            model_filename = 'best_random_forest_model.onnx'
            s3_key = f"models/{model_filename}"
            s3_client.head_object(Bucket=s3_bucket, Key=s3_key)

            # Si el modelo está disponible, lo carga
            model_bytes = load_onnx_model_from_minio(model_filename, 'train_random_forest_to_minio')
            random_forest_session = ort.InferenceSession(model_bytes)

            input_name = random_forest_session.get_inputs()[0].name
            output_name = random_forest_session.get_outputs()[0].name
            inputs = {input_name: features}
            prediction = random_forest_session.run([output_name], inputs)[0]

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Si el modelo aún no está disponible en S3, usa el modelo por defecto
                input_name = random_forest_default_session.get_inputs()[0].name
                output_name = random_forest_default_session.get_outputs()[0].name
                inputs = {input_name: features}
                prediction = random_forest_default_session.run([output_name], inputs)[0]
            else:
                raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    result = "Yes" if prediction[0] == 1 else "No"
    
    return {"prediction": result}




@app.get("/check_dag_status/{dag_id}")
async def check_dag_status(dag_id: str):
    session = SessionLocal()
    dag_table = Table('dag', metadata, autoload_with=engine)
    dag_run_table = Table('dag_run', metadata, autoload_with=engine)
    try:
        # Verificar existencia de DAG
        dag_query = select(dag_table).where(dag_table.c.dag_id == dag_id)
        dag_result = session.execute(dag_query).first()
        if not dag_result:
            raise HTTPException(status_code=404, detail=f"DAG with id '{dag_id}' not found")

        dag_run_query = (
            select(dag_run_table)
            .where(dag_run_table.c.dag_id == dag_id)
            .order_by(dag_run_table.c.execution_date.desc())
            .limit(1)
        )
        dag_run_result = session.execute(dag_run_query).first()

        if not dag_run_result:
            raise HTTPException(status_code=404, detail=f"No runs found for DAG with id '{dag_id}'")

        run = {
            "run_id": dag_run_result.dag_run_id,
            "state": dag_run_result.state,
            "execution_date": dag_run_result.execution_date
        }

        # Verificar existencia de la tabla rf_metrics
        inspector = inspect(session.get_bind())
        if not inspector.has_table("rf_metrics"):
            raise HTTPException(status_code=404, detail="Table 'rf_metrics' not found in the database.")

        # Verificar existencia de la tabla weather_data
        if not inspector.has_table("weather_data"):
            raise HTTPException(status_code=404, detail="Table 'weather_data' not found in the database.")

        return {"dag_id": dag_id, "last_run": run }

    finally:
        session.close()

@app.get('/check_model_status/{model_filename}')
async def check_model_status(model_filename: str): 
    # Verificar si el modelo está disponible en S3
    s3_key = f"models/{model_filename}"
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        model_from_minio = True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail=f"Model '{model_filename}' is not available in S3 yet.")
        else:
            raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")
    
    return {"message": f"Model '{model_filename}' is available in S3.", "model_from_minio": model_from_minio }

   
######### OBTENER METRICAS ##########

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    try:
        rf_metrics = Table(
        'rf_metrics', metadata,
        autoload_with=engine
        )
        # fetch the latest metrics entry
        stmt = select(
            rf_metrics.c.accuracy_train,
            rf_metrics.c.accuracy_test,
            rf_metrics.c.confusion_matrix,
            rf_metrics.c.f1_score,
            rf_metrics.c.roc_auc,
            rf_metrics.c.fpr,
            rf_metrics.c.tpr
        ).order_by(rf_metrics.c.id.desc()).limit(1)

        result = db.execute(stmt).fetchone()

        if result:
            # Convert confusion matrix from string to list
            confusion_matrix_str = result.confusion_matrix
            print(f"Confusion matrix string from DB: {confusion_matrix_str}")  # Debugging print

            try:
                # Attempt to parse confusion matrix as JSON
                confusion_matrix = json.loads(confusion_matrix_str)

                # Validate that confusion matrix is a list of two lists
                if not isinstance(confusion_matrix, list) or len(confusion_matrix) != 2:
                    raise ValueError("Confusion matrix does not have exactly 2 rows.")

                # Assuming it's a binary classification, extract TN, FP, FN, TP
                tn, fp = confusion_matrix[0]  # First list corresponds to TN, FP
                fn, tp = confusion_matrix[1]  # Second list corresponds to FN, TP

                # Prepare the response JSON
                metrics_data = {
                    'accuracy_train': result.accuracy_train,
                    'accuracy_test': result.accuracy_test,
                    'confusion_matrix': confusion_matrix,
                    'f1_score': result.f1_score,
                    'roc_auc': result.roc_auc,
                    'fpr': result.fpr if result.fpr else [],
                    'tpr': result.tpr if result.tpr else []
                }

                return metrics_data

            except (ValueError, json.JSONDecodeError) as e:
                raise HTTPException(status_code=500, detail=f"Error parsing confusion matrix: {str(e)}")

        else:
            raise HTTPException(status_code=404, detail="No metrics found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@app.get("/year-most-rain")
async def get_year_most_rain(db: Session = Depends(get_db)):
    try:
        weather_data = Table(
            'weather_data', metadata,
            autoload_with=engine
        )

        # Obtener el año con más lluvia
        stmt = (
            select(
                weather_data.c.Year,
                func.sum(weather_data.c.Rainfall).label('total_rainfall')
            )
            .group_by(weather_data.c.Year)
            .order_by(desc('total_rainfall'))
            .limit(1)
        )

        result = db.execute(stmt).fetchone()

        if result:
            return {
                'year': result.Year,
                'total_rainfall': result.total_rainfall
            }
        else:
            raise HTTPException(status_code=404, detail="No weather data found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/season-most-rain")
async def get_season_most_rain(db: Session = Depends(get_db)):
    try:
        weather_data = Table(
        'weather_data', metadata,
        autoload_with=engine
        )

        # Obtener la estación del año donde más llovió
        stmt = select(
            weather_data.c.Season,
            func.sum(weather_data.c.Rainfall).label('total_rainfall')
        ).group_by(weather_data.c.Season).order_by(desc('total_rainfall')).limit(1)

        result = db.execute(stmt).fetchone()

        if result:
            # Obtener el rango de años cubiertos por los datos meteorológicos
            stmt_years = select(
                func.min(weather_data.c.Year).label('min_year'),
                func.max(weather_data.c.Year).label('max_year')
            )
            years_result = db.execute(stmt_years).fetchone()

            if years_result:
                min_year = years_result.min_year
                max_year = years_result.max_year
                year_range = f"{min_year} - {max_year}"
            else:
                year_range = "No data available"

            return {
                'season': result.Season,
                'total_rainfall': result.total_rainfall,
                'year_range': year_range
            }
        else:
            raise HTTPException(status_code=404, detail="No weather data found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/city-most-rain")
async def get_city_most_rain(db: Session = Depends(get_db)):
    try:
        weather_data = Table(
            'weather_data', metadata,
            autoload_with=engine
        )

        # Obtener la ciudad donde más llovió
        stmt = select(
            weather_data.c.Location,
            func.sum(weather_data.c.Rainfall).label('total_rainfall')
        ).group_by(weather_data.c.Location).order_by(desc('total_rainfall')).limit(1)

        result = db.execute(stmt).fetchone()

        if result:
            # Obtener el rango de años cubiertos por los datos meteorológicos
            stmt_years = select(
                func.min(weather_data.c.Year).label('min_year'),
                func.max(weather_data.c.Year).label('max_year')
            )
            years_result = db.execute(stmt_years).fetchone()

            if years_result:
                min_year = years_result.min_year
                max_year = years_result.max_year
                year_range = f"{min_year} - {max_year}"
            else:
                year_range = "No data available"

            return {
                'location': result.Location,
                'total_rainfall': result.total_rainfall,
                'year_range': year_range
            }
        else:
            raise HTTPException(status_code=404, detail="No weather data found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/average-rainfall-last-5-years")
async def get_average_rainfall_last_5_years(db: Session = Depends(get_db)):
    try:
        weather_data = Table(
            'weather_data', metadata,
            autoload_with=engine
        )

        # Obtener el promedio de lluvia por año para los últimos 5 años
        last_year_query = select(func.max(weather_data.c.Year))
        last_year = db.execute(last_year_query).scalar()
        stmt = select(
            weather_data.c.Year,
            func.avg(weather_data.c.Rainfall).label('avg_rainfall')
        ).where(weather_data.c.Year >= (last_year - 4)).group_by(weather_data.c.Year).order_by(desc(weather_data.c.Year))

        results = db.execute(stmt).fetchall()

        if results:
            average_rainfall = [{
                'year': row.Year,
                'avg_rainfall': row.avg_rainfall
            } for row in results]
            
            total_rainfall = sum([year['avg_rainfall'] for year in average_rainfall])
            # Obtener el rango de años
            first_year = results[-1].Year
            last_year = results[0].Year

            return {
                'average_rainfall': average_rainfall,
                'year_range': f"{first_year} - {last_year}",
                'total_rainfall': total_rainfall
            }

        else:
            raise HTTPException(status_code=404, detail="No weather data found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/trainings-by-date")
async def get_trainings_by_date(db: Session = Depends(get_db)):
    try:
        rf_metrics = Table(
            'weather_data', metadata,
            autoload_with=engine
        )

        # Obtener todos los entrenamientos agrupados por fecha
        stmt = (
            select(
                func.date_trunc('day', rf_metrics.c.created_at).label('training_date'),
                func.avg(rf_metrics.c.accuracy_train).label('avg_accuracy_train'),
                func.avg(rf_metrics.c.accuracy_test).label('avg_accuracy_test'),
                func.avg(rf_metrics.c.f1_score).label('avg_f1_score'),
                func.avg(rf_metrics.c.roc_auc).label('avg_roc_auc')
            )
            .group_by(func.date_trunc('day', rf_metrics.c.created_at))
            .order_by(func.date_trunc('day', rf_metrics.c.created_at).desc())
        )

        results = db.execute(stmt).fetchall()

        if results:
            trainings = [{
                'training_date': row.training_date.date().isoformat(),
                'avg_accuracy_train': row.avg_accuracy_train,
                'avg_accuracy_test': row.avg_accuracy_test,
                'avg_f1_score': row.avg_f1_score,
                'avg_roc_auc': row.avg_roc_auc
            } for row in results]

            return trainings

        else:
            raise HTTPException(status_code=404, detail="No training data found")

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
