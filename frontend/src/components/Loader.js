import React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import './Loader.css';

const Loader = () => {
  return (
    <div className="loader-container">
      <CircularProgress color="inherit" />
    </div>
  );
};

export default Loader;
