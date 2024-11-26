# Depression Prediction for Kaggle Playground Series - Season 4, Episode 11

This project focuses on predicting depression based on a set of mental health survey data as part of the Kaggle competition [Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11/overview). Using demographic details, mental health history, and lifestyle habits, this project preprocesses the data and applies a machine learning model to generate predictions.

---

## Problem Statement

The dataset provided includes mental health survey data, with features like demographic information, mental health history, and lifestyle habits. The goal is to predict whether an individual suffers from depression based on the provided features.

### Evaluation Metric
The competition uses `F1-Score` as the evaluation metric for predictions on the test set.

---

## Dataset Overview

The dataset contains the following:
- **Training Data (`train.csv`)**: Includes features and target labels indicating the presence of depression.
- **Test Data (`test.csv`)**: Includes similar features, but the target label is missing and needs to be predicted.

### Key Features
- **Demographic Details**: `Gender`, `City`, `Profession`, etc.
- **Lifestyle Habits**: `Sleep Duration`, `Dietary Habits`, etc.
- **Mental Health History**: `Family History of Mental Illness`, `Have you ever had suicidal thoughts?`, etc.

---

## Features of the Project

1. **Data Preprocessing:**
   - Combines related features like `Work Pressure` and `Academic Pressure` into a single feature.
   - Handles missing values:
     - Replaces missing values in numeric columns with `0.0`.
     - Replaces missing values in non-numeric columns with `'Unknown'`.
   - Encodes categorical features using `LabelEncoder`.
   - Groups rare categories in features like `City`, `Profession`, and `Degree` into `"Other"`.
   - Drops unnecessary columns like `Name` and `CGPA`.

2. **Machine Learning Model:**
   - A **Random Forest Classifier** is used to train and predict depression.
   - The model is evaluated using metrics such as accuracy, F1-Score, and more.

3. **Submission File:**
   - Generates a `submission.csv` file with predictions for the test dataset.

---

## Installation

### Prerequisites
- Python 3.x
- Required libraries:
  ```bash
  pandas
  numpy
  scikit-learn
