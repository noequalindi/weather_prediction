import React from 'react';
import { Line } from 'react-chartjs-2';
import { Card, CardContent, Box } from '@mui/material';

const ROCGraph = ({ metricsData }) => {
  const { fpr, tpr, roc_auc: rocAuc } = metricsData;

  // Convertir fpr y tpr de strings JSON a arrays
  const parsedFpr = JSON.parse(fpr);
  const parsedTpr = JSON.parse(tpr);

  const data = {
    labels: parsedFpr,  // Usar fpr como labels si ya es un array válido
    datasets: [
      {
        label: `ROC Curve (AUC = ${rocAuc.toFixed(2)})`,
        data: parsedTpr.map((tprValue, index) => ({ x: parsedFpr[index], y: tprValue })),
        fill: false,
        borderColor: 'rgba(75,192,192,1)',
        tension: 0.4,
        cubicInterpolationMode: 'default',
      },
      {
        label: 'Random Classifier (AUC = 0.5)',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        fill: false,
        borderColor: 'rgba(255, 99, 132, 1)',
        borderDash: [5, 5],
        tension: 0,
      },
    ],
  };

  const options = {
    scales: {
      x: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'False Positive Rate',
        },
      },
      y: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'True Positive Rate',
        },
      },
    },
  };

  return (
    <Card sx={{ borderRadius: 2, backgroundColor: 'white', p: 2, color: 'black' }}>
      <CardContent>
         <h3> ROC Curve</h3>
         <Box sx={{ height: '320px', alignContent: 'center', display: 'flex', justifyContent: 'center' }}>
          <Line data={data} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default ROCGraph;
