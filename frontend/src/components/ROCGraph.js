import React from 'react';
import { Line } from 'react-chartjs-2';
import { Card, CardContent, Box } from '@mui/material';

const ROCGraph = ({ metricsData }) => {
  const { fpr, tpr, roc_auc: rocAuc } = metricsData;

  // Convertir fpr y tpr de strings JSON a arrays
  const parsedFpr = JSON.parse(fpr);
  const parsedTpr = JSON.parse(tpr);

  // Asegurarse de que los datos estén ordenados por fpr
  const sortedData = parsedFpr.map((fprValue, index) => ({
    fpr: fprValue,
    tpr: parsedTpr[index],
  })).sort((a, b) => a.fpr - b.fpr);

  const data = {
    datasets: [
      {
        label: `ROC curve (area = ${rocAuc.toFixed(4)})`,
        data: sortedData.map(d => ({ x: d.fpr, y: d.tpr })),
        fill: false,
        borderColor: 'rgba(75,192,192,1)', // Mantener color original
        tension: 0.4,
        cubicInterpolationMode: 'default',
      },
      {
        label: 'Random Classifier (AUC = 0.5)',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        fill: false,
        borderColor: 'rgba(255, 99, 132, 1)', // Mantener color original
        borderDash: [5, 5],
        tension: 0,
      },
    ],
  };

  const options = {
    plugins: {
      legend: {
        labels: {
          font: {
            size: 16, // Tamaño de la fuente para las etiquetas de la leyenda
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'Tasa de Falsos Positivos',
          font: {
            size: 18, // Tamaño de la fuente para el título del eje x
          },
        },
        ticks: {
          stepSize: 0.1, // Incremento de 0.1 en el eje x
          font: {
            size: 14, // Tamaño de la fuente para las etiquetas del eje x
          },
        },
      },
      y: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'Tasa de Verdaderos Positivos',
          font: {
            size: 18, // Tamaño de la fuente para el título del eje y
          },
        },
        ticks: {
          stepSize: 0.1, // Incremento de 0.1 en el eje y
          font: {
            size: 14, // Tamaño de la fuente para las etiquetas del eje y
          },
        },
      },
    },
  };

  return (
    <Card sx={{ borderRadius: 2, backgroundColor: 'white', p: 2, color: 'black' }}>
      <CardContent>
        <h3 style={{ fontSize: '1.6rem' }}>Curvas ROC</h3> {/* Tamaño de la fuente para el título */}
        <Box sx={{ height: '320px', alignContent: 'center', display: 'flex', justifyContent: 'center' }}>
          <Line data={data} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default ROCGraph;
