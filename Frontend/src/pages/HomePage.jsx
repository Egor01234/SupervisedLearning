import React from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';
import accidentsByHourImg from '../assets/Plots/accidents_by_hour.png';
import accidentClustersImg from '../assets/Plots/accident_clusters.png';
import accidentLocationsImg from '../assets/Plots/accident_locations.png';
import confusionDTImg from '../assets/Plots/confusion_matrix_DecisionTree.png';
import confusionGBImg from '../assets/Plots/confusion_matrix_GradientBoosting.png';
import confusionLRImg from '../assets/Plots/confusion_matrix_LogisticRegression.png';
import confusionNBImg from '../assets/Plots/confusion_matrix_NaiveBayes.png';
import confusionRFImg from '../assets/Plots/confusion_matrix_RandomForest.png';
import confusionSVMImg from '../assets/Plots/confusion_matrix_SVM.png';
import correlationHeatmapImg from '../assets/Plots/correlation_heatmap.png';
import fatalVsNonFatalImg from '../assets/Plots/fatal_vs_non_fatal_accidents.png';
import modelPerformanceImg from '../assets/Plots/model_performance_comparison.png';

const plotsToShow = [
  { src: fatalVsNonFatalImg, title: 'Fatal vs. Non-Fatal Accidents' },
  { src: accidentsByHourImg, title: 'Accidents by Hour' },
  { src: correlationHeatmapImg, title: 'Feature Correlation Heatmap' },
  { src: modelPerformanceImg, title: 'Model Performance Comparison' },
  { src: accidentLocationsImg, title: 'Accident Locations Map' },
  { src: accidentClustersImg, title: 'Accident Clusters (DBSCAN)' },
  { src: confusionDTImg, title: 'Confusion Matrix: Decision Tree' },
  { src: confusionRFImg, title: 'Confusion Matrix: Random Forest' },
  { src: confusionGBImg, title: 'Confusion Matrix: Gradient Boosting' },
  { src: confusionSVMImg, title: 'Confusion Matrix: SVM' },
  { src: confusionLRImg, title: 'Confusion Matrix: Logistic Regression' },
  { src: confusionNBImg, title: 'Confusion Matrix: Naive Bayes' },
];


function HomePage() {
  return (
    <>
      <Header />
      <div className="App home-page">
        <h2>Welcome!</h2>
        <p>
          This tool predicts the severity (Fatal vs. Non-Fatal Injury) of motorcyclist accidents
          based on various factors entered by the user.
        </p>

        <div className="report-summary">
          <h3>Project Analysis Highlights:</h3>
          <ul>
            <li>Analyzed a dataset of <strong>1684 motorcyclist accidents</strong>.</li>
            <li>The data showed a significant class imbalance, with only <strong>~13.5%</strong> of cases being fatal. This was addressed using techniques like SMOTE during model training.</li>
            <li>Several machine learning models were evaluated, including Decision Tree, Random Forest, Gradient Boosting, SVM, and Logistic Regression.</li>
            <li>The <strong>Decision Tree model demonstrated the best overall performance</strong> on the test data, achieving high accuracy (94.7%) and effectively identifying fatal accidents (Recall: 87%).</li>
            <li>The trained models are now available for you to use for prediction.</li>
          </ul>
        </div>

        <div className="plots-section">
          <h3>Data Visualizations</h3>
          <p>Key visualizations generated during the analysis:</p>
          <div className="plots-grid">
            {plotsToShow.map((plot, index) => (
              <div key={index} className="plot-item">
                <img src={plot.src} alt={plot.title} />
                <p className="plot-title">{plot.title}</p>
              </div>
            ))}
          </div>
        </div>

        <p>
          Click the button below to select a prediction model and input accident details to see the predicted severity.
        </p>

        <Link to="/select-model" className="button-link">
          <button className="get-started-btn">Get Started</button>
        </Link>
      </div>
    </>
  );
}

export default HomePage;