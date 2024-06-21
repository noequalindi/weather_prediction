import React, { useState, useEffect } from 'react';
import ConfusionMatrixGraph from './ConfusionMatrixGraph';
import MetricsTable from './MetricsTable';
import ROCGraph from './ROCGraph';

const Metrics = ({ metrics }) => {
    if (!metrics) {
        return null; // o algún indicador de carga
      }
  return (
    <div className="metrics-container">
      <h2 className="metrics-title">Random Forest Model Metrics</h2>
      {metrics && (
        <div>
          <ROCGraph metricsData={metrics} />
          <ConfusionMatrixGraph metricsData={metrics} />
          <MetricsTable metrics={metrics} />
        </div>
      )}
    </div>
  );
};

export default Metrics;
