import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ModelSelectionPage from './pages/ModelSelectionPage';
import PredictionPage from './pages/PredictionPage';
import './App.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/select-model" element={<ModelSelectionPage />} />
      <Route path="/predict/:modelName" element={<PredictionPage />} />
    </Routes>
  );
}

export default App;