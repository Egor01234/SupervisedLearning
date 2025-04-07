import React, { useState } from 'react';
import motorcycleHeaderImage from './assets/Motorcycle.jpg';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [formData, setFormData] = useState({
    ACCLOC: '',
    MANOEUVER: '',
    AUTOMOBILE: '',
    STREET1: '',
    TIME: '',
    RDSFCOND: ''
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);
    const predictEndpoint = `${API_URL}/predict`;

    try {
      const dataToSend = {
        ...formData,
        AUTOMOBILE: parseInt(formData.AUTOMOBILE, 10) || 0
      };

      if (isNaN(dataToSend.AUTOMOBILE)) {
        throw new Error("Automobile count must be a valid number.");
      }

      const response = await fetch(predictEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend)
      });

      if (!response.ok) {
        let errorData = { message: `API Error: ${response.status} ${response.statusText}` };
        try {
          errorData = await response.json();
        } catch (parseError) {
           // Ignore
        }
        throw new Error(errorData.error || errorData.message);
      }

      const result = await response.json();
      setPredictionResult(result);

    } catch (err) {
      console.error("Prediction request failed:", err);
      setError(err.message || 'Failed to fetch prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatProbability = (prob) => {
    if (typeof prob !== 'number') return 'N/A';
    return (prob * 100).toFixed(2) + '%';
  };

  return (
    <>
      <div className="page-header">
        <img
          src={motorcycleHeaderImage}
          alt="Various types of motorcycles"
          className="header-image"
        />
        <h1 className="main-title">Motorcyclist Accident Severity Prediction</h1>
      </div>

      <div className="App">
        <p className="example-text">
          Example Input: Location=Intersection, Manoeuver=Turning Left, Automobile=1, Street=Main St, Time=1545, Road Condition=Dry
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="ACCLOC">Accident Location Type:</label>
            <input
              type="text"
              id="ACCLOC"
              name="ACCLOC"
              value={formData.ACCLOC}
              onChange={handleChange}
              required
              placeholder="e.g., Intersection, Non Intersection"
            />
          </div>

          <div className="form-group">
            <label htmlFor="MANOEUVER">Manoeuver:</label>
            <input
              type="text"
              id="MANOEUVER"
              name="MANOEUVER"
              value={formData.MANOEUVER}
              onChange={handleChange}
              required
              placeholder="e.g., Going Ahead, Turning Left"
            />
          </div>

          <div className="form-group">
             <label htmlFor="AUTOMOBILE">Involves Automobile (count):</label>
             <input
               type="number"
               id="AUTOMOBILE"
               name="AUTOMOBILE"
               value={formData.AUTOMOBILE}
               onChange={handleChange}
               min="0"
               required
               placeholder="e.g., 0, 1, 2"
             />
           </div>

          <div className="form-group">
            <label htmlFor="STREET1">Street Name:</label>
            <input
              type="text"
              id="STREET1"
              name="STREET1"
              value={formData.STREET1}
              onChange={handleChange}
              required
              placeholder="e.g., Yonge St, Queen St W"
            />
          </div>

          <div className="form-group">
            <label htmlFor="TIME">Time (HHMM format):</label>
            <input
              type="text"
              id="TIME"
              name="TIME"
              value={formData.TIME}
              onChange={handleChange}
              required
              placeholder="e.g., 0830, 1700"
            />
          </div>

          <div className="form-group">
            <label htmlFor="RDSFCOND">Road Surface Condition:</label>
            <input
              type="text"
              id="RDSFCOND"
              name="RDSFCOND"
              value={formData.RDSFCOND}
              onChange={handleChange}
              required
              placeholder="e.g., Dry, Wet, Slush"
            />
          </div>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Predicting...' : 'Predict Severity'}
          </button>
        </form>

        {isLoading && <div className="loading">Loading...</div>}

        {error && <div className="error">Error: {error}</div>}

        {predictionResult && !error && (
          <div className="result">
            <h2>Prediction Result:</h2>
            <p className={`prediction ${predictionResult.prediction_code === 1 ? 'fatal' : 'non-fatal'}`}>
              {predictionResult.prediction}
            </p>
            <p>Probability (Fatal): {formatProbability(predictionResult.probability_fatal)}</p>
            <p>Probability (Non-Fatal): {formatProbability(predictionResult.probability_non_fatal)}</p>
          </div>
        )}
      </div>
    </>
  );
}

export default App;