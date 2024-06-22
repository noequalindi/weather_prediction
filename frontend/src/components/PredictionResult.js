import React from 'react';
import './PredictionResult.css';
import rainIcon from '../img/rain.png';
import sunnyIcon from '../img/sun.png';

const PredictionResult = ({ prediction }) => {
  const showRainIcon = prediction && prediction === 'Yes'; // Cambiar a la condición correcta según tu respuesta de la API

  return (
    <div className="prediction-result">
        <>
          <h2>Resultado</h2>
          <p>Lloverá?: {prediction}</p>
          <img
            className="weather-icon"
            src={showRainIcon ? rainIcon : sunnyIcon}
            alt={showRainIcon ? 'Rain Icon' : 'Sunny Icon'}
          />
        </>
    </div>
  );
};

export default PredictionResult;
