# Car Price Predictor

Welcome to the Car Price Predictor! This project uses machine learning to estimate the price of a used car based on its features such as make, model, year, kilometers driven, and fuel type.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)

## Introduction

This web application allows users to predict the price of a used car by entering relevant details. The backend is powered by a machine learning model, specifically an XGBoost regressor, which has been fine-tuned using GridSearchCV for optimal performance.

## Features

- **User-friendly Interface**: Simple and intuitive UI built with Bootstrap.
- **Machine Learning Model**: Uses XGBoost regressor for price prediction.
- **Data Preprocessing**: Handles various data preprocessing tasks including handling missing values, encoding categorical features, and data cleaning.
- **Model Tuning**: The model is fine-tuned using GridSearchCV to find the best hyperparameters.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/prince41773/Car-price-prediction-using-XGBoost-regression-algorithm.git
    cd Car-price-prediction-using-XGBoost-regression-algorithm
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the dataset (`car_data.csv`) in the project directory.

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Enter the car details in the form and click the "Predict Price" button to get the estimated price.

## Model Training

If you want to retrain the model, you can use the provided script in `app.py`:

1. Ensure you have the dataset (`car_data.csv`) in the project directory.
2. Run the script to train the model:
    ```bash
    python app.py
    ```
3. The trained model will be saved as `BestXGBoostModelWithRegularization.pkl`.
