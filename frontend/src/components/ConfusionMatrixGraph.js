import React from 'react';
import { Chart, registerables } from 'chart.js';
import { BarController, BarElement } from 'chart.js';
import { Chart as ChartJS } from 'react-chartjs-2';
import { Box } from '@mui/material';

Chart.register(...registerables);
Chart.register(BarController, BarElement);

const ConfusionMatrixGraph = ({ metricsData }) => {
  if (!metricsData || !metricsData.confusion_matrix) {
    return (
      <Box sx={{ backgroundColor: 'white', borderRadius: 2, p: 2 }}>
        <h2>Confusion Matrix</h2>
        <Box>
          <p>No data available</p>
        </Box>
      </Box>
    );
  }

  const { confusion_matrix } = metricsData;

  // Ensure confusion matrix is 2x2
  if (confusion_matrix.length !== 2 || confusion_matrix[0].length !== 2 || confusion_matrix[1].length !== 2) {
    console.error('Invalid confusion matrix structure.');
    return null; // or handle the error appropriately
  }

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

  const options = {
    scales: {
      x: {
        stacked: true,
        title: {
          display: true,
          text: 'True Label',
        },
        ticks: {
          // Customizing x axis labels
          callback: (value) => (value === 0 ? 'Yes' : 'No'),
        }
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
          title: () => 'Confusion Matrix',
          label: (tooltipItem) => {
            const datasetLabel = tooltipItem.dataset.label;
            const value = tooltipItem.raw;
            return `${datasetLabel}: ${value}`;
          },
        },
      },
    },
  };

  return (
    <Box sx={{ backgroundColor: 'white', borderRadius: 2, p: 2, color: 'black' }}>
      <h3>Matríz de Confusión</h3>
      <Box>
        <ChartJS type='bar' data={data} options={options} />
      </Box>
    </Box>
  );
};

export default ConfusionMatrixGraph;
