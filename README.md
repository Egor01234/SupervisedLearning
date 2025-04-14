# Motorcyclist Accident Severity Prediction
# Use these Imports to run Backend

## pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn geopandas contextily joblib Flask Flask-CORS

## or 
## conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn imbalanced-learn geopandas contextily joblib flask flask-cors
# And for Frontend
## npm install react react-dom react-router-dom
This project predicts the severity (Fatal vs. Non-Fatal Injury) of motorcyclist accidents based on various input factors. It consists of a Python Flask backend API for model training and prediction, and a React frontend for user interaction.

## Table of Contents

*   [Overview](#overview)
*   [Features](#features)
*   [Technology Stack](#technology-stack)
*   [Prerequisites](#prerequisites)
*   [Installation and Setup](#installation-and-setup)
    *   [Backend (Flask API)](#backend-flask-api)
    *   [Frontend (React UI)](#frontend-react-ui)
*   [Running the Application](#running-the-application)
    *   [Backend](#backend)
    *   [Frontend](#frontend)
*   [Usage](#usage)
*   [Project Structure](#project-structure)
*   [Troubleshooting](#troubleshooting)
*   [License](#license)

## Overview

The application analyzes historical motorcyclist accident data to train several machine learning models. Users can select a trained model via the web interface, input details about a hypothetical accident scenario, and receive a prediction about the likely severity, including probabilities.

## Features

*   **Data Analysis & Visualization:** Explores the dataset and generates plots (saved in `backend/Plots`).
*   **Model Training:** Trains multiple classification models (Decision Tree, Random Forest, Gradient Boosting, SVM, Logistic Regression) using scikit-learn.
*   **Preprocessing Pipeline:** Implements robust preprocessing including imputation, scaling, and one-hot encoding.
*   **Class Imbalance Handling:** Uses SMOTE and class weighting to address imbalanced data.
*   **Model Evaluation & Comparison:** Compares model performance using various metrics.
*   **API:** Flask-based REST API to serve predictions using saved models.
*   **Interactive Frontend:** React-based user interface for selecting models and inputting data.
*   **Prediction Results:** Displays prediction (Fatal/Non-Fatal) and associated probabilities.

## Technology Stack

*   **Backend:**
    *   Python 3.x
    *   Flask (Web Framework)
    *   Pandas (Data Manipulation)
    *   NumPy (Numerical Operations)
    *   Scikit-learn (Machine Learning)
    *   Imbalanced-learn (SMOTE)
    *   Joblib (Model Persistence)
    *   Flask-CORS (Cross-Origin Resource Sharing)
    *   GeoPandas, Matplotlib, Seaborn, Contextily (for EDA/Visualization in `model.py`)
*   **Frontend:**
    *   React
    *   JavaScript (ES6+)
    *   HTML5 / CSS3
    *   React Router (`react-router-dom`)
    *   Fetch API (for backend communication)
*   **Development:**
    *   Node.js & npm / yarn (for frontend)
    *   Python Virtual Environments (`venv`)

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.9 or higher recommended. [Download Python](https://www.python.org/downloads/)
*   **pip:** Python package installer (usually comes with Python).
*   **Node.js:** Version 18.x or higher recommended. [Download Node.js](https://nodejs.org/)
*   **npm** (comes with Node.js) or **yarn** (optional package manager).

## Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

### Backend (Flask API)

1.  **Navigate to Backend Directory:**
    ```bash
    cd backend
    ```

2.  **Create and Activate Virtual Environment:** (Highly Recommended)
    *   **Windows (cmd/powershell):**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux (bash/zsh):**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *(Why use a venv? It isolates project dependencies, preventing conflicts with other Python projects or your global installation.)*

3.  **Install Requirements:**
    *   **Crucial Step:** Ensure you have a `requirements.txt` file in the `backend` directory that was generated from the **exact Python environment where the models were originally trained** (especially matching the `scikit-learn` version, e.g., 1.4.2 based on recent issues). If missing, generate it from the training environment using `pip freeze > requirements.txt`.
    *   Install the dependencies:
        ```bash
        pip install -r requirements.txt
        ```

4.  **(Optional) Train Models:**
    *   The application expects pre-trained model files (`.pkl`) in the `backend/Models_plk/` directory.
    *   If these files are missing, or you want to retrain the models using the provided data (`MOTORCYCLIST_KSI_...csv`), run the training script:
        ```bash
        python model.py
        ```
    *   This script will perform EDA, train models, evaluate them, save plots to `backend/Plots/`, and save the trained model pipelines to `backend/Models_plk/`. Ensure you run this using the **correct Python environment** (matching the versions specified in `requirements.txt`).

### Frontend (React UI)

1.  **Navigate to Frontend Directory:**
    (Assuming it's one level up from `backend` and then into `frontend`. Adjust if necessary)
    ```bash
    cd ../frontend
    ```
    *Or, from the project root:*
    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**
    *   Using npm:
        ```bash
        npm install
        ```
    *   Or using yarn:
        ```bash
        yarn install
        ```

3.  **Configure API URL (Optional):**
    *   The frontend needs to know where the backend API is running. It expects an environment variable `VITE_API_URL`.
    *   Create a file named `.env` in the `frontend` directory.
    *   Add the following line, adjusting the URL if your backend runs on a different port or host:
        ```env
        VITE_API_URL=http://localhost:5000
        ```
    *   If this file is omitted, the frontend might default to `http://localhost:5000`.

## Running the Application

You need to run both the backend and frontend servers simultaneously.

### Backend

1.  **Navigate to Backend Directory:**
    ```bash
    cd path/to/your/project/backend
    ```
2.  **Activate Virtual Environment:**
    *   **Windows:** `.\venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`
3.  **Start the Flask Server:**
    ```bash
    python app.py
    ```
    The API should now be running, typically at `http://localhost:5000` (or `http://127.0.0.1:5000`). Observe the console output for successful model loading messages or errors.

### Frontend

1.  **Navigate to Frontend Directory:**
    ```bash
    cd path/to/your/project/frontend
    ```
2.  **Start the React Development Server:**
    *   Using npm:
        ```bash
        npm run dev
        ```
    *   Or using yarn:
        ```bash
        yarn dev
        ```
    The frontend development server will start, usually on a port like `3000` or `5173`. Your browser might open automatically, or you can navigate to the URL shown in the terminal (e.g., `http://localhost:5173`).

## Usage

1.  Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`).
2.  You will see the **Homepage** with project information and visualizations.
3.  Click **"Get Started"**.
4.  On the **Model Selection Page**, choose one of the available machine learning models.
5.  You will be taken to the **Prediction Page**. Fill in all the required input fields describing the accident scenario.
6.  Click the **"Predict with [Model Name]"** button.
7.  The frontend will send the data to the backend API. After a moment, the predicted severity (Fatal or Non-Fatal Injury) and the associated probabilities will be displayed below the form.
8.  You can go back to select a different model or modify the inputs for another prediction.

## Project Structure

```plaintext
<repository-root>/
├── backend/
│   ├── Models_plk/             # Saved scikit-learn model pipelines (.pkl)
│   │   ├── DecisionTree_model.pkl
│   │   └── ... (other models)
│   ├── Plots/                  # Saved plots from model.py EDA
│   │   ├── accidents_by_hour.png
│   │   └── ... (other plots)
│   ├── venv/                   # Python virtual environment (if created)
│   ├── app.py                  # Flask application (API)
│   ├── model.py                # Model training and evaluation script
│   ├── requirements.txt        # Backend Python dependencies
│   └── MOTORCYCLIST_KSI_...csv # Dataset (ensure it's present if needed)
│
├── frontend/
│   ├── public/                 # Static assets
│   ├── src/
│   │   ├── assets/             # Frontend assets (like images from Plots/)
│   │   ├── components/         # Reusable React components (e.g., Header)
│   │   ├── pages/              # Page-level components (HomePage, PredictionPage, etc.)
│   │   ├── App.css             # Main CSS styling
│   │   ├── App.js              # Main application component with routing
│   │   └── index.js/main.jsx   # Entry point for React app
│   ├── .env                    # Environment variables (e.g., VITE_API_URL) - Optional
│   ├── package.json            # Frontend dependencies and scripts
│   └── ... (other config files like vite.config.js, etc.)
│
└── README.md                   # This file
