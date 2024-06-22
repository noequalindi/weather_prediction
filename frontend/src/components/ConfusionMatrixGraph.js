import React, { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import { Box } from '@mui/material';

Chart.register(...registerables);

const ConfusionMatrixGraph = ({ metricsData }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!metricsData || !metricsData.confusion_matrix) {
      return;
    }

    const { confusion_matrix } = metricsData;

    // Ensure confusion matrix is 2x2
    if (confusion_matrix.length !== 2 || confusion_matrix[0].length !== 2 || confusion_matrix[1].length !== 2) {
      console.error('Invalid confusion matrix structure.');
      return;
    }

    // Prepare data for chart
    const data = {
      labels: ['True - Yes', 'True - No'],
      datasets: [
        {
          label: 'Predicted - Yes',
          data: [confusion_matrix[0][0], confusion_matrix[1][0]],
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
        },
        {
          label: 'Predicted - No',
          data: [confusion_matrix[0][1], confusion_matrix[1][1]],
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
      ],
    };

    // Chart options
    const options = {
      scales: {
        x: {
          stacked: true,
          title: {
            display: true,
            text: 'True Label',
          },
        },
        y: {
          stacked: true,
          title: {
            display: true,
            text: 'Count',
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: (tooltipItems) => '',
            label: (tooltipItem) => {
              const datasetLabel = tooltipItem.dataset.label || '';
              const value = tooltipItem.raw || '';
              return `${datasetLabel}: ${value}`;
            },
          },
        },
      },
    };

    if (chartRef.current) {
      const ctx = chartRef.current.getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: data,
        options: options,
      });
    }

  }, [metricsData]);

  return (
    <Box sx={{ backgroundColor: 'white', borderRadius: 2, p: 2, color: 'black', marginTop: 2 }}>
      <h3>Gráfico de barras apiladas</h3>
      <canvas ref={chartRef}></canvas>
    </Box>
  );
};

export default ConfusionMatrixGraph;
