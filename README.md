# Depression Prediction for Kaggle Playground Series - Season 4, Episode 11

This project is part of the Kaggle competition [Playground Series - Season 4, Episode 11](https://www.kaggle.com/competitions/playground-series-s4e11/overview). The goal is to predict whether a person has depression based on survey data.

---

## Project Overview

### **Dataset**
- **Train Data:** Contains survey features and labels (`Depression` column).
- **Test Data:** Contains survey features without labels (used for predictions).

### **Steps in the Project**
1. **Data Preprocessing:**
   - Combined related columns (e.g., `Work Pressure` + `Academic Pressure` → `Pressure`).
   - Replaced missing values:
     - `'Unknown'` for non-numeric columns.
     - `0.0` for numeric columns.
   - Encoded categorical data using `LabelEncoder`.
   - Grouped rare categories into `"Other"` for features like `City`, `Profession`, and `Degree`.
   - Dropped unnecessary columns (`Name` and `CGPA`).

2. **Machine Learning:**
   - Used a **Random Forest Classifier** to predict depression.
   - Evaluated the model using accuracy and ROC AUC scores.

3. **Submission File:**
   - Generated a `submission.csv` with predictions for Kaggle.

---

## Usage

### **Run the Project**
1. Place `train.csv` and `test.csv` in the project folder.
2. Run the script:
   ```bash
   python main_V2.py
   
The script will:
Train a model on the training data.
Predict labels for the test data.
Create a submission.csv file for Kaggle.

File Structure:
├── train.csv          # Training data
├── test.csv           # Test data
├── main.py            # Script for preprocessing, training, and prediction
├── main_V1.py            # Script for preprocessing, training, and prediction
├── main_V2.py            # Script for preprocessing, training, and prediction
├── submission.csv     # Generated predictions for Kaggle
└── README.md          # Project documentation

Results
Accuracy: Displays the percentage of correct predictions.
Submission Format: The submission.csv file looks like this:

id,Depression
1,0
2,1
3,0

Acknowledgements
Kaggle: For hosting the competition.
Scikit-learn: For tools to preprocess data and train models.

How to Contribute
If you’d like to improve this project:
Fork the repository.
Create a feature branch:
bash
git checkout -b feature-branch
Push your changes and create a Pull Request.

License
This project is licensed under the MIT License.



### **What to Do Next**
- Copy and use this simplified version for your project.
- Add or remove any parts you feel are unnecessary.
- If you don’t understand something in the README, ask, and I’ll clarify or simplify further.
