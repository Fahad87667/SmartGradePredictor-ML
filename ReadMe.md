# SmartGradePredictor-ML

SmartGradePredictor-ML is a machine learning project that predicts student grades based on the number of courses taken and study hours. The project utilizes various regression models to analyze and forecast academic performance.

## Overview

This project demonstrates the process of building, training, and evaluating machine learning models for predicting student grades. It covers data loading, preprocessing, model training, and performance evaluation.

## Features

- Predict student grades using machine learning techniques.
- Evaluate multiple regression models: Linear Regression, Lasso, Ridge, and ElasticNet.
- Save and load the best-performing model for future predictions.

## Dataset

The project uses a dataset (`Student_Marks.csv`) containing information about the number of courses, study hours, and corresponding marks. Missing values are handled by filling them with the mean value of the respective columns.

## Getting Started

1. Clone the repository:

````bash
git clone https://github.com/yourusername/SmartGradePredictor-ML.git

2. Install the required dependencies:

```bash
pip install -r requirements.txt

3. Run the project:

```bash
python app.py


## Models

The project trains and evaluates the following regression models:

- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression

The best-performing model is saved using joblib for future predictions.

## Evaluation Metrics

The models are evaluated based on the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2) Score

## Future Enhancements

- Explore additional features for improved predictions.
- Experiment with advanced machine learning techniques.
- Develop a user interface for easy interaction.


````
