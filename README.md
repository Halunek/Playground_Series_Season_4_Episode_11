# Depression Prediction for Kaggle Playground Series - Season 4, Episode 11

This project is a machine learning solution for the Kaggle competition [Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11/overview). The goal of this competition is to predict depression based on a set of mental health survey data. The dataset includes various features like demographic information, mental health history, and lifestyle habits.

## Problem Statement

Mental health is an important factor in well-being, yet it is often underrepresented in discussions around health. The competition provides survey data to analyze the patterns and factors contributing to depression and build a predictive model for it.

## Dataset

The dataset consists of:
- **Training Data:** Includes features like demographic details, mental health history, and labels indicating whether the respondent suffers from depression (`Depression` column).
- **Test Data:** Includes similar features without the target label, used for generating predictions.
- **Evaluation Metric:** Submissions are evaluated using `Accuracy Score`.

More details about the dataset and the problem can be found on the [competition page](https://www.kaggle.com/competitions/playground-series-s4e11/overview).

---

## Features

1. **Data Preprocessing:**
   - Handles missing values by replacing them with relevant defaults (`Unknown` for non-numeric columns and `0.0` for numeric columns).
   - Encodes categorical data using `LabelEncoder`.
   - Processes rare categories in columns like `Profession`, `City`, and `Degree` by grouping them into "Other".

2. **Machine Learning Model:**
   - Uses a **Random Forest Classifier** for training and prediction.
   - Trains the model using the processed training dataset.
   - Evaluates the model on validation data using metrics like:
     - Accuracy
     - ROC AUC
     - F1-Score
     - Confusion Matrix

3. **Output:**
   - Generates a `submission.csv` file containing predictions for the test dataset.

