import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Box, Typography } from '@mui/material';

const apiURL = 'http://localhost:8000';

const MetricsTable = ({ metrics }) => {
  const [trainingsByDate, setTrainingsByDate] = useState([]);
  const [averageRainfallLast5Years, setAverageRainfallLast5Years] = useState([]);
  const [cityMostRain, setCityMostRain] = useState(null);
  const [seasonMostRain, setSeasonMostRain] = useState(null);
  const [yearMostRain, setYearMostRain] = useState(null);

  const [trainingsByDatePeriod, setTrainingsByDatePeriod] = useState(null);
  const [averageRainfallLast5YearsPeriod, setAverageRainfallLast5YearsPeriod] = useState(null);
  const [cityMostRainPeriod, setCityMostRainPeriod] = useState(null);
  const [seasonMostRainPeriod, setSeasonMostRainPeriod] = useState(null);
  const [yearMostRainPeriod, setYearMostRainPeriod] = useState(null);
  const [totalRainfall, setTotalRainfall] = useState(0);

  const calculatePercentage = (value, total) => (total > 0 ? (value / total) * 100 : 0).toFixed(2);

  const calculatePercentageMetrics = (value, total) => {
    const decimalFormatted = value.toFixed(4); 
    const percentage = (value / total * 100).toFixed(0); 
    return `${decimalFormatted} - ${percentage}%`;
  };

  const fontStyle = {fontSize: '1rem', fontFamily: 'Poppins'}
  useEffect(() => {
    const fetchTrainingsByDate = async () => {
      try {
        const response = await axios.get(`${apiURL}/trainings-by-date`);
        if (response.status === 200) {
          setTrainingsByDate(response.data);
          setTrainingsByDatePeriod(response.data.year_range); // Establecer el período desde y hasta
        } else {
          console.error('Failed to fetch trainings by date');
        }
      } catch (error) {
        console.error('Error fetching trainings by date:', error);
      }
    };

    const fetchAverageRainfallLast5Years = async () => {
      try {
        const response = await axios.get(`${apiURL}/average-rainfall-last-5-years`);
        if (response.status === 200) {
          setAverageRainfallLast5Years(response.data.average_rainfall);
          setAverageRainfallLast5YearsPeriod(response.data.year_range); // Establecer el período desde y hasta
          setTotalRainfall(response.data.total_rainfall);
        } else {
          console.error('Failed to fetch average rainfall last 5 years');
        }
      } catch (error) {
        console.error('Error fetching average rainfall last 5 years:', error);
      }
    };

    const fetchCityMostRain = async () => {
      try {
        const response = await axios.get(`${apiURL}/city-most-rain`);
        if (response.status === 200) {
          setCityMostRain(response.data);
          setCityMostRainPeriod(response.data.year_range); // Establecer el período desde y hasta
        } else {
          console.error('Failed to fetch city with most rain');
        }
      } catch (error) {
        console.error('Error fetching city with most rain:', error);
      }
    };

    const fetchSeasonMostRain = async () => {
      try {
        const response = await axios.get(`${apiURL}/season-most-rain`);
        if (response.status === 200) {
          setSeasonMostRain(response.data);
          setSeasonMostRainPeriod(response.data.year_range); // Establecer el período desde y hasta
        } else {
          console.error('Failed to fetch season with most rain');
        }
      } catch (error) {
        console.error('Error fetching season with most rain:', error);
      }
    };

    const fetchYearMostRain = async () => {
      try {
        const response = await axios.get(`${apiURL}/year-most-rain`);
        if (response.status === 200) {
          setYearMostRain(response.data);
          setYearMostRainPeriod(response.data.year_range); // Establecer el período desde y hasta
        } else {
          console.error('Failed to fetch year with most rain');
        }
      } catch (error) {
        console.error('Error fetching year with most rain:', error);
      }
    };

    fetchTrainingsByDate();
    fetchAverageRainfallLast5Years();
    fetchCityMostRain();
    fetchSeasonMostRain();
    fetchYearMostRain();

  }, []);

  return (
    <Box>
      <h2>Otras Métricas</h2>

    <TableContainer component={Paper} sx={{ backgroundColor: 'white', borderRadius: 2, marginBottom: 2, fontFamily: 'Poppins', fontSize: '1.2rem' }}>
    <h3>Métricas del último entrenamiento del modelo Random Forest</h3>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell sx={fontStyle}>Métrica</TableCell>
            <TableCell sx={fontStyle}>Train</TableCell>
            <TableCell sx={fontStyle}>Test</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell sx={fontStyle}>Accuracy</TableCell>
            <TableCell sx={fontStyle}>{calculatePercentageMetrics(metrics.accuracy_train, 1)}</TableCell>
            <TableCell sx={fontStyle}>{calculatePercentageMetrics(metrics.accuracy_test, 1)}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={fontStyle}>F1 Score</TableCell>
            <TableCell sx={fontStyle}>{calculatePercentageMetrics(metrics.f1_score, 1)}</TableCell> {/* Ajustar para mantener 3 celdas */}
            <TableCell sx={fontStyle}></TableCell> {/* Ajustar para mantener 3 celdas */}
          </TableRow>
          <TableRow>
            <TableCell sx={fontStyle}>ROC AUC</TableCell>
            <TableCell colSpan={2} sx={fontStyle}>{metrics.roc_auc.toFixed(4)}</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>

      {/* Métricas de entrenamientos por fecha */}
      {trainingsByDate.length > 0 && (
        <Box mb={4}>
     
          <TableContainer component={Paper} sx={{ marginBottom: 2, borderRadius: '10px' }}>
          <h4>Entrenamientos por Fecha </h4>

            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={fontStyle}>Fecha</TableCell>
                  <TableCell sx={fontStyle}>Promedio Accuracy Train</TableCell>
                  <TableCell sx={fontStyle}>Promedio Accuracy Test</TableCell>
                  <TableCell sx={fontStyle}>Promedio F1 Score</TableCell>
                  <TableCell sx={fontStyle}>Promedio ROC AUC</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {trainingsByDate.map(training => (
                  <TableRow key={training.training_date}>
                    <TableCell sx={fontStyle}>{training.training_date}</TableCell>
                    <TableCell sx={fontStyle}>{calculatePercentageMetrics(training.avg_accuracy_train, 1)}</TableCell>
                    <TableCell sx={fontStyle}>{calculatePercentageMetrics(training.avg_accuracy_test, 1)}</TableCell>
                    <TableCell sx={fontStyle}>{calculatePercentageMetrics(training.avg_f1_score,1)}</TableCell>
                    <TableCell sx={fontStyle}>{training.avg_roc_auc.toFixed(4)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
      {/* Promedio de lluvia en los últimos 5 años */}
    <Box mb={4}>
      <h2> Más sobre el Dataset AUS_WEATHER </h2>
    {averageRainfallLast5Years.length > 0 && (
      <Box mb={4}>
        <h3>Promedio de Lluvia en los Últimos 5 Años ({averageRainfallLast5YearsPeriod})</h3>
        <TableContainer component={Paper} sx={{...fontStyle, borderRadius: 2 }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell align="center" sx={fontStyle}>Año</TableCell>
                <TableCell align="center" sx={fontStyle}>Promedio de Lluvia (mm)</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {averageRainfallLast5Years.map(year => (
                <TableRow key={year.year}>
                  <TableCell align="center" sx={fontStyle}>{year.year}</TableCell>
                  <TableCell align="center" sx={fontStyle}>{calculatePercentage(year.avg_rainfall, totalRainfall)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    )}

      {/* Ciudad donde más llovió */}
      {cityMostRain && (
        <Box mb={4}>
          <h3>Ciudad donde Más Llovió ({cityMostRainPeriod})</h3>
          <Paper elevation={3} sx={{ p: 2, borderRadius: 2 }}>
            <Typography sx={fontStyle}><strong>Ciudad:</strong> {cityMostRain.location}</Typography>
            <Typography sx={fontStyle}><strong>Total de Lluvia (mm):</strong> {cityMostRain.total_rainfall.toFixed(2) / 100}</Typography>
          </Paper>
        </Box>
      )}

      {/* Estación del año donde más llovió */}
      {seasonMostRain  && (
        <Box mb={4}>
          <h3>Estación del Año donde Más Llovió ({seasonMostRainPeriod})</h3>
          <Paper elevation={3} sx={{ p: 2, borderRadius: 2 }}>
            <Typography sx={fontStyle}><strong>Estación:</strong> {seasonMostRain.season}</Typography>
            <Typography sx={fontStyle}><strong>Total de Lluvia (mm):</strong> {seasonMostRain.total_rainfall.toFixed(2)}</Typography>
          </Paper>
        </Box>
      )}

      {/* Año con más lluvia */}
      {yearMostRain && (
        <Box mb={4}>
          <h3 >Año con Más Lluvia</h3>
          <Paper elevation={3} sx={{ p: 2, borderRadius: 2 }}>
            <Typography sx={fontStyle}><strong>Año:</strong> {yearMostRain.year}</Typography>
            <Typography sx={fontStyle}><strong>Total de Lluvia (mm):</strong> {yearMostRain.total_rainfall.toFixed(4)}</Typography>
          </Paper>
        </Box>
      )}
    </Box>
  </Box>
  );
};

export default MetricsTable;
