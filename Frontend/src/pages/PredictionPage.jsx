import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import Header from '../components/Header';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const formatModelName = (key) => {
    switch (key) {
        case 'decisiontree': return 'Decision Tree';
        case 'randomforest': return 'Random Forest';
        case 'gradientboosting': return 'Gradient Boosting';
        case 'logisticregression': return 'Logistic Regression';
        case 'svm': return 'Support Vector Machine (SVM)';
        default: return key;
    }
};

function PredictionPage() {
  const { modelName } = useParams();
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

  useEffect(() => {
      setFormData({
        ACCLOC: '', MANOEUVER: '', AUTOMOBILE: '',
        STREET1: '', TIME: '', RDSFCOND: ''
      });
      setPredictionResult(null);
      setError(null);
      setIsLoading(false);
  }, [modelName]);

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
    const predictEndpoint = `${API_URL}/predict/${modelName}`;

    try {
      const dataToSend = {
        ...formData,
        AUTOMOBILE: formData.AUTOMOBILE,
      };

      if (!formData.ACCLOC || !formData.MANOEUVER || !formData.AUTOMOBILE || !formData.STREET1 || !formData.TIME || !formData.RDSFCOND) {
          throw new Error("All fields are required.");
      }
      if (isNaN(parseInt(formData.AUTOMOBILE, 10)) || parseInt(formData.AUTOMOBILE, 10) < 0) {
           throw new Error("Automobile count must be a non-negative number.");
      }
      if (!/^\d{3,4}$/.test(formData.TIME) || parseInt(formData.TIME.padStart(4,'0').substring(0,2)) > 23 || parseInt(formData.TIME.padStart(4,'0').substring(2,4)) > 59) {
          throw new Error("Time must be in HHMM format (e.g., 0830, 1700).");
      }

      const response = await fetch(predictEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend)
      });

      const resultData = await response.json();

      if (!response.ok) {
        throw new Error(resultData.error || `API Error: ${response.status} ${response.statusText}`);
      }

      setPredictionResult(resultData);

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

  const displayModelName = formatModelName(modelName);

  return (
    <>
      <Header title={`Predict Severity using ${displayModelName}`} />
      <div className="App">
        <p className="example-text">
          Fill in the details below to predict accident severity using the <strong>{displayModelName}</strong> model.
        </p>

        <form onSubmit={handleSubmit}>
           <div className="form-group">
             <label htmlFor="ACCLOC">Accident Location Type:</label>
             <input type="text" id="ACCLOC" name="ACCLOC" value={formData.ACCLOC} onChange={handleChange} required placeholder="e.g., Intersection, Non Intersection" />
           </div>
           <div className="form-group">
             <label htmlFor="MANOEUVER">Manoeuver:</label>
             <input type="text" id="MANOEUVER" name="MANOEUVER" value={formData.MANOEUVER} onChange={handleChange} required placeholder="e.g., Going Ahead, Turning Left" />
           </div>
           <div className="form-group">
              <label htmlFor="AUTOMOBILE">Involves Automobile (count):</label>
              <input type="number" id="AUTOMOBILE" name="AUTOMOBILE" value={formData.AUTOMOBILE} onChange={handleChange} min="0" required placeholder="e.g., 0, 1, 2" />
            </div>
           <div className="form-group">
             <label htmlFor="STREET1">Street Name:</label>
             <input type="text" id="STREET1" name="STREET1" value={formData.STREET1} onChange={handleChange} required placeholder="e.g., Yonge St, Queen St W" />
           </div>
           <div className="form-group">
             <label htmlFor="TIME">Time (HHMM format):</label>
             <input type="text" id="TIME" name="TIME" value={formData.TIME} onChange={handleChange} required placeholder="e.g., 0830, 1700" pattern="\d{3,4}" title="Enter time as 3 or 4 digits (e.g., 830 or 1700)"/>
           </div>
           <div className="form-group">
             <label htmlFor="RDSFCOND">Road Surface Condition:</label>
             <input type="text" id="RDSFCOND" name="RDSFCOND" value={formData.RDSFCOND} onChange={handleChange} required placeholder="e.g., Dry, Wet, Slush" />
           </div>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Predicting...' : `Predict with ${displayModelName}`}
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

        <Link to="/select-model" className="button-link back-button-link">
            <button className="back-button">Select Different Model</button>
        </Link>
      </div>
    </>
  );
}

export default PredictionPage;