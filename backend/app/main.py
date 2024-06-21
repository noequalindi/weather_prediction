from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import onnxruntime as ort
import ast
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sklearn.metrics import roc_curve, roc_auc_score
import boto3
import botocore
from io import BytesIO
import json

load_dotenv()
app = FastAPI()

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

def load_onnx_model_from_minio(model_filename):
    s3_bucket = "weather-prediction-s3"
    s3_key = f"models/{model_filename}"

    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        model_bytes = response['Body'].read()
        return model_bytes
    except botocore.exceptions.EndpointConnectionError as e:
        error_message = f"Could not connect to the endpoint URL: {MINIO_ENDPOINT_URL}/{s3_bucket}/{s3_key}"
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {error_message}")
    except botocore.exceptions.ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {str(e)}")

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

running_in_docker = os.getenv('RUNNING_IN_DOCKER', 'False') == 'True'

# Rutas de los modelos ONNX
if running_in_docker:
    scaler_model_path = '../models/standard_scaler_model.onnx'
    decision_tree_model_path = '../models/best_decision_tree_model.onnx'
else:
    scaler_model_path = os.getenv('MODELS_PATH') + 'standard_scaler_model.onnx'
    decision_tree_model_path = os.getenv('MODELS_PATH') + 'best_decision_tree_model.onnx'

# Crear sesiones de inferencia para ambos modelos
scaler_session = ort.InferenceSession(scaler_model_path)
decision_tree_session = ort.InferenceSession(decision_tree_model_path)
random_forest_model = load_onnx_model_from_minio('best_random_forest_model.onnx')
random_forest_session = ort.InferenceSession(random_forest_model)

@app.post("/predict/{model_type}")
async def predict(data: RainPrediction, model_type: str = "decision_tree"):
    if scaler_session is None or decision_tree_session is None : # or random_forest_session is None
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = np.array([[data.MinTemp, data.Rainfall, data.WindSpeed9am, data.Humidity3pm, data.RainToday]], dtype=np.float32)

    if model_type == "decision_tree":
        # Inferencia con el modelo de DecisionTreeClassifier
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
        # Inferencia con el modelo de RandomForestClassifier
        input_name = random_forest_session.get_inputs()[0].name
        output_name = random_forest_session.get_outputs()[0].name
        inputs = {input_name: features}
        prediction = random_forest_session.run([output_name], inputs)[0]

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    result = "Yes" if prediction[0] == 1 else "No"
    
    return {"prediction": result}


######### OBTENER METRICAS ##########

DATABASE_URL = 'postgresql+psycopg2://airflow:airflow@localhost:5432/airflow'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Definir la tabla rf_metrics
metadata = MetaData()
rf_metrics = Table(
    'rf_metrics', metadata,
    autoload_with=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Definir la tabla rf_metrics
rf_metrics = Table(
    'rf_metrics', metadata,
    autoload_with=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Example function to calculate ROC curve
def calculate_roc_curve(y_true, y_pred_proba):
    # Replace with actual ROC curve calculation logic
    return [], [], 0.0

# Example function to calculate ROC curve
def calculate_roc_curve(y_true, y_pred_proba):
    # Replace with actual ROC curve calculation logic
    return [], [], 0.0

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    try:
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

                # Calculate fpr, tpr for ROC curve (example placeholders)
                y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
                y_pred_proba = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)  # Example prediction, replace with actual probabilities

                fpr, tpr, roc_auc = calculate_roc_curve(y_true, y_pred_proba)

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