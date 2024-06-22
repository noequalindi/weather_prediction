// PredictionForm.js

import React, { useState } from 'react';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import {
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  RadioGroup,
  Radio,
  FormControlLabel,
  FormLabel,
  Grid,
  Box,
  Typography,
  FormHelperText
} from '@mui/material';
import './PredictionForm.css';
import PredictionResult from './PredictionResult';

const baseURL = 'http://localhost:8000';

const PredictionForm = () => {
  const [features, setFeatures] = useState({
    MinTemp: '',
    Rainfall: '',
    WindSpeed9am: '',
    Humidity3pm: '',
    RainToday: ''
  });
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('decision_tree'); // Opción por defecto
  const [error, setError] = useState(false); // Estado para manejar errores
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFeatures({
      ...features,
      [name]: value,
    });
  };

  const handleClearFields = () => {
    setFeatures({
      MinTemp: '',
      Rainfall: '',
      WindSpeed9am: '',
      Humidity3pm: '',
      RainToday: ''
    });
    setErrorMessage(''); // Limpiar el mensaje de error
    setPrediction(null); // Limpiar el resultado de la predicción
    setError(false); // Reiniciar el estado de error
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const validateInput = () => {
    const { MinTemp, Rainfall, WindSpeed9am, Humidity3pm, RainToday } = features;
    if (
      isNaN(parseFloat(MinTemp)) ||
      isNaN(parseFloat(Rainfall)) ||
      isNaN(parseFloat(WindSpeed9am)) ||
      isNaN(parseFloat(Humidity3pm)) ||
      isNaN(parseInt(RainToday))
    ) {
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateInput()) {
      setErrorMessage('* Todos los campos son obligatorios y deben ser numéricos.');
      return;
    }
    setErrorMessage('');
    setLoading(true);
    try {
      const response = await axios.post(baseURL + `/predict/${selectedModel}`, {
        MinTemp: parseFloat(features.MinTemp),
        Rainfall: parseFloat(features.Rainfall),
        WindSpeed9am: parseFloat(features.WindSpeed9am),
        Humidity3pm: parseFloat(features.Humidity3pm),
        RainToday: parseInt(features.RainToday)
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        withCredentials: true,
        timeout: 5000
      });
      setPrediction(response.data.prediction);
      setError(false); // Reiniciar el estado de error si la solicitud fue exitosa
    } catch (error) {
      console.error("There was an error making the request:", error);
      if (error.response) {
        // El servidor devolvió un código de estado diferente de 2xx
        setErrorMessage(error.response.data.detail);
      } else if (error.request) {
        // La solicitud fue hecha pero no se recibió respuesta
        setErrorMessage("No se recibió respuesta del servidor.");
      } else {
        // Ocurrió un error antes de realizar la solicitud
        setErrorMessage(error.message);
      }
      setError(true); // Establecer estado de error en true si ocurre un error
    } finally {
      setTimeout(() => setLoading(false), 2000); // Detener el loader después de 2 segundos
    }
  };


  return (
    <div className='prediction-container'>
      <form className="prediction-form" onSubmit={handleSubmit}>
        <div className="form-grid">
          <label className="input-label">
            Temperatura Mínima:
            <span className='description'>en grados centígrados</span>
            <TextField
              type="text"
              name="MinTemp"
              value={features.MinTemp}
              onChange={handleChange}
              variant="outlined"
              className="input-field"
              fullWidth
              required
            />
          </label>
          <label className="input-label">
            Lluvia en mm:
            <TextField
              type="text"
              name="Rainfall"
              value={features.Rainfall}
              onChange={handleChange}
              variant="outlined"
              className="input-field"
              fullWidth
              required
            />
          </label>
          <label className="input-label">
            Velocidad del viento:
            <span className='description'>en km/h</span>
            <TextField
              type="text"
              name="WindSpeed9am"
              value={features.WindSpeed9am}
              onChange={handleChange}
              variant="outlined"
              className="input-field"
              fullWidth
              required
            />
          </label>
          <label className="input-label">
            Humedad:
            <TextField
              type="text"
              name="Humidity3pm"
              value={features.Humidity3pm}
              onChange={handleChange}
              variant="outlined"
              className="input-field"
              fullWidth
              required
            />
          </label>
          <FormControl component="fieldset" className="radio-container">
            <FormLabel component="legend">Indicar si llovió hoy:</FormLabel>
            <RadioGroup
              row
              aria-label="RainToday"
              name="RainToday"
              value={features.RainToday}
              onChange={handleChange}
            >
              <FormControlLabel
                value="1"
                control={<Radio />}
                label="Sí"
                className="radio-label"
              />
              <FormControlLabel
                value="0"
                control={<Radio />}
                label="No"
                className="radio-label"
              />
            </RadioGroup>
          </FormControl>
          <FormControl className="model-select">
            <InputLabel id="model-select-label">Seleccione el modelo</InputLabel>
            <Select
              labelId="model-select-label"
              id="model-select"
              value={selectedModel}
              onChange={handleModelChange}
              label="Seleccione el modelo"
              variant="outlined"
              fullWidth
              required
            >
              <MenuItem value="decision_tree">Árbol de Decisión</MenuItem>
              <MenuItem value="random_forest">Random Forest</MenuItem>
            </Select>
          </FormControl>
        </div>
        {errorMessage && <div className="error-message">{errorMessage}</div>}
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disableElevation
          fullWidth
          className="submit-button"
          sx={{ fontSize: "1.2rem", fontFamily: 'Poppins', marginTop: '1rem', borderRadius: '50px' }}
        >
         Predict 
        </Button>
      </form>
      <Button
        variant="contained"
        color="text"
        disableElevation
        fullWidth
        className="clear-button"
        onClick={handleClearFields}
        sx={{ fontSize: "1rem", fontFamily: 'Poppins', marginTop: '1rem', borderRadius: '50px' }}
      >
        Limpiar Campos
      </Button>
      {error ? (
        <div className="error-message">
          <h3>Ocurrió un error al hacer la predicción.</h3>
        </div>
      ) : ( 
      loading ? <CircularProgress color="primary" className="loader" /> :
      !loading && prediction && <PredictionResult prediction={prediction} error={error} />
    )}
    </div>
  );
};

export default PredictionForm;
