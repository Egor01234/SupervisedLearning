import React from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';

const models = [
  { name: "Decision Tree", key: "decisiontree" },
  { name: "Random Forest", key: "randomforest" },
  { name: "Gradient Boosting", key: "gradientboosting" },
  { name: "Logistic Regression", key: "logisticregression" },
  { name: "Support Vector Machine (SVM)", key: "svm" },
];

function ModelSelectionPage() {
  return (
    <>
      <Header title="Select Prediction Model" />
      <div className="App model-selection-page">
        <h2>Choose a Model</h2>
        <p>Select one of the following machine learning models to perform the prediction:</p>
        <div className="model-buttons">
          {models.map((model) => (
            <Link key={model.key} to={`/predict/${model.key}`} className="button-link">
              <button>{model.name}</button>
            </Link>
          ))}
        </div>
         <Link to="/" className="button-link back-button-link">
            <button className="back-button">Back to Home</button>
        </Link>
      </div>
    </>
  );
}

export default ModelSelectionPage;