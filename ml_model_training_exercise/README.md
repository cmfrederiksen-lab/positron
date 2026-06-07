# Modeltræning og Evaluering: Clinical Diagnostic Pipeline

**Author:** Casper M. Frederiksen

## Overview

This repository contains a test machine learning pipeline designed to evaluate and compare classification models on the Breast Cancer Wisconsin (Diagnostic) dataset. 

This project implements strict software engineering principles including configuration management, data schema validation, and systematic experiment tracking.

## Core Architecture
* **Validation:** [Pydantic](https://docs.pydantic.dev/) is used to strictly type and validate all model hyperparameters and data split ratios before execution.
* **Modeling:** [Scikit-learn](https://scikit-learn.org/) powers the training and evaluation of a Random Forest Classifier and a Logistic Regression model (utilizing the `liblinear` solver for optimized performance on binary tasks).
* **Tracking:** [MLflow](https://mlflow.org/) is integrated to automatically log all Pydantic configurations, track performance metrics (Accuracy, Precision, Recall), and serialize the trained model artifacts.

## Project Structure
```text
model_training_exercise/
├── data/
│   ├── raw/                # Immutable original data (if loading externally)
│   └── processed/          # Cleaned, structured data
├── src/
│   ├── config/
│   │   └── settings.py     # Pydantic configuration schemas
│   ├── data/
│   │   └── loader.py       # Data ingestion pipeline
│   └── models/
│       └── train.py        # Core training and evaluation loop
├── .gitignore
├── README.md
└── requirements.txt

