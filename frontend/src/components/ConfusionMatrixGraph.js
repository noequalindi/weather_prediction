import React, { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import { Box } from '@mui/material';
import { MatrixController, MatrixElement } from 'chartjs-chart-matrix'; // Importa los módulos necesarios
import './Charts.css'
Chart.register(...registerables, MatrixController, MatrixElement);

const ConfusionMatrixGraph = ({ metricsData }) => {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

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
      datasets: [{
        label: 'Confusion Matrix',
        data: [
          { x: 'Predicted Yes', y: 'True No', v: confusion_matrix[1][0], color: 'rgba(255, 99, 132, 0.5)' }, // Predicted Yes, True No (rojo)
          { x: 'Predicted No', y: 'True No', v: confusion_matrix[1][1], color: 'rgba(75, 192, 192, 0.5)' }, // Predicted No, True No (verde)
          { x: 'Predicted Yes', y: 'True Yes', v: confusion_matrix[0][0], color: 'rgba(75, 192, 192, 0.5)' }, // Predicted Yes, True Yes (verde)
          { x: 'Predicted No', y: 'True Yes', v: confusion_matrix[0][1], color: 'rgba(255, 99, 132, 0.5)'}, // Predicted No, True Yes (rojo)
        ],
        backgroundColor: context => {
          return context.dataset.data[context.dataIndex].color;
        },
        borderColor: 'rgba(0, 0, 0, 1)',
        borderWidth: 1,
        fontSize: 18,
        width: () => 100, // Ajusta el tamaño de la celda
        height: () => 100, // Ajusta el tamaño de la celda
      }]
    };

    // Chart options
    const options = {
      plugins: {
        tooltip: {
          callbacks: {
            label: context => {
              const { x, y, v } = context.raw;
              return `True: ${y}, Predicted: ${x}, Count: ${v}`;
            },
          },
          title: {
            font: {
              size: 18, // Ajusta el tamaño de la tipografía para el título del tooltip
            },
          },
        },
        legend: {
          labels: {
            font: {
              size: 16, // Ajusta el tamaño de la tipografía para las leyendas
            },
          },
        },
      },
      scales: {
        x: {
          type: 'category',
          labels: ['Predicted Yes', 'Predicted No'],
          title: {
            display: true,
            text: 'Predicted Label',
            font: {
              size: 18, // Ajusta el tamaño de la tipografía para el título del eje x
            },
          },
        },
        y: {
          type: 'category',
          labels: ['True No', 'True Yes'], // Invertido: True No arriba, True Yes abajo
          title: {
            display: true,
            text: 'True Label',
            font: {
              size: 18, // Ajusta el tamaño de la tipografía para el título del eje y
            },
          },
        },
      },
    };

    if (chartRef.current) {
      const ctx = chartRef.current.getContext('2d');

      // Destruye el gráfico anterior si existe
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }

      // Crea un nuevo gráfico y guárdalo en la referencia
      chartInstanceRef.current = new Chart(ctx, {
        type: 'matrix',
        data: data,
        options: options,
      });
    }
  }, [metricsData]);

  // Limpiar el gráfico cuando el componente se desmonta
  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);

  return (
    <Box sx={{ backgroundColor: 'white', borderRadius: 2, p: 2, color: 'black', marginTop: 2}}>
      <h3 style={{ fontSize: '1.5rem' }}>Confusion Matrix</h3>
      <canvas ref={chartRef}></canvas>
    </Box>
  );
};

export default ConfusionMatrixGraph;
