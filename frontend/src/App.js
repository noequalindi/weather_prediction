import React, { useState, useEffect } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import weatherImg from './img/weather-icon.png';
import Metrics from './components/Metrics';
import Loader from './components/Loader'; // Importar el componente Loader
import axios from 'axios';

const API_URL = 'http://localhost:8000';
const DAG_ID = 'train_random_forest_to_minio'; 

function App() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true); 
  const [loadingMessage, setLoadingMessage] = useState("Se está entrenando el modelo en background...");
  const [dagLoaded, setDagLoaded] = useState(false); 

  useEffect(() => {
    const checkDagStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/check_dag_status/${DAG_ID}`);
        if (response.status === 200) {
          const lastRun = response.data.last_run;
          if (lastRun && lastRun.state === 'success') {
            setDagLoaded(true);
            fetchMetrics();
          } else {
            setTimeout(checkDagStatus, 5000);
          }
        }
      } catch (error) {
        if (error.response && error.response.status === 404) {
          setLoadingMessage('El modelo se está cargando, espere unos segundos...');
        } else {
          console.error('Error al verificar el estado del DAG:', error);
          setLoadingMessage('Error al verificar el estado del DAG.');
        }
        setTimeout(checkDagStatus, 5000); // Volver a intentar después de 5 segundos para cualquier error
      }
    };

    const fetchMetrics = async () => {
      try {
        const response = await axios.get(`${API_URL}/metrics`);
        setMetrics(response.data);
        setLoading(false); // Al recibir la respuesta, se oculta el loader
      } catch (error) {
        console.error('Error al obtener métricas:', error);
        setLoading(false); // Manejar errores y ocultar el loader
      }
    };

    checkDagStatus();
  }, []); 

  return (
    <div className="App poppins-semibold">
      <header className="App-header">
        <div className="logo-title">
          <h1>Rain Prediction</h1>
          <img className="weather-icon" src={weatherImg} alt="Weather Icon" />
        </div>
         <PredictionForm />
        {loading ? (
          <>
            <Loader />
            <h4>{loadingMessage}</h4>
          </>
        ) : (
          <>
            {dagLoaded && metrics && <Metrics metrics={metrics} />}
          </>
        )}
      </header>
    </div>
  );
}

export default App;
