import React, { useState, useEffect } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import weatherImg from './img/weather-icon.png';
import Metrics from './components/Metrics';
import Loader from './components/Loader'; // Importar el componente Loader
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true); // Estado para controlar la carga

  useEffect(() => {
    axios.get(`${API_URL}/metrics`)
      .then(response => {
        setMetrics(response.data);
        setLoading(false); // Al recibir la respuesta, se oculta el loader
      })
      .catch(error => {
        console.error('Error fetching metrics:', error);
        setLoading(false); // Manejar errores y ocultar el loader
      });
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
          <><Loader />
            <h4>Cargando Métricas...</h4></>
        ) : (
          metrics && <Metrics metrics={metrics} />
        )}
      </header>
    </div>
  );
}

export default App;
