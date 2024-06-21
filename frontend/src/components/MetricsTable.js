import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';

const MetricsTable = ({ metrics }) => {
  return (
    <div>
      <h2>Otras Métricas</h2>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              <TableCell>Train</TableCell>
              <TableCell>Test</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Accuracy</TableCell>
              <TableCell>{metrics.accuracy_train}</TableCell>
              <TableCell>{metrics.accuracy_test}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>F1 Score</TableCell>
              <TableCell>{metrics.f1_score}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>ROC AUC</TableCell>
              <TableCell colSpan={2}>{metrics.roc_auc}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default MetricsTable;
