import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print("First rows of train csv file:")
print(df_train.head())
print("\nColumn names and data type of each column:")
print(df_train.dtypes)
print("\nCheck missing values in each column:")
print(df_train.isnull().sum())

# Combine Work pressure and Academic pressure columns into Pressure and drop them:
df_train['Pressure'] = df_train['Work Pressure'].fillna(df_train['Academic Pressure'])
df_test['Pressure'] = df_test['Work Pressure'].fillna(df_test['Academic Pressure'])
df_train.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)
df_test.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)

# Combine Study Satisfaction and Job Satisfaction columns into Satisfaction and drop them:
df_train['Satisfaction'] = df_train['Study Satisfaction'].fillna(df_train['Job Satisfaction'])
df_test['Satisfaction'] = df_test['Study Satisfaction'].fillna(df_test['Job Satisfaction'])
df_train.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)
df_test.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

# Combine Working Professional or Student and Profession columns into Profession and drop Working Professional or
# Student:
df_train['Profession'] = np.where(df_train['Working Professional or Student'] == 'Student', 'Student',
                                  df_train['Profession'])
df_test['Profession'] = np.where(df_test['Working Professional or Student'] == 'Student', 'Student',
                                 df_test['Profession'])

df_train.drop(['Working Professional or Student'], axis=1, inplace=True)
df_test.drop(['Working Professional or Student'], axis=1, inplace=True)

df_test['Profession'] = df_test['Profession'].fillna("Other")
df_train['Profession'] = df_train['Profession'].fillna("Other")

# Mapping of two choice columns:
# Map binary columns:
binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

# Apply mapping:
df_train['Gender'] = df_train['Gender'].map(binary_mapping)
df_test['Gender'] = df_test['Gender'].map(binary_mapping)

df_train['Have you ever had suicidal thoughts ?'] = df_train['Have you ever had suicidal thoughts ?'].map(
    binary_mapping)
df_test['Have you ever had suicidal thoughts ?'] = df_test['Have you ever had suicidal thoughts ?'].map(binary_mapping)

df_train['Family History of Mental Illness'] = df_train['Family History of Mental Illness'].map(binary_mapping)
df_test['Family History of Mental Illness'] = df_test['Family History of Mental Illness'].map(binary_mapping)

# Define mapping for Dietary habits:
dietary_mapping = {
    'Moderate': 1,
    'Unhealthy': 2,
    'Healthy': 3,
}

df_train['Dietary Habits'] = df_train['Dietary Habits'].map(dietary_mapping).fillna(0)
df_test['Dietary Habits'] = df_test['Dietary Habits'].map(dietary_mapping).fillna(0)

# Define the mapping for specific categories for Sleep
sleep_mapping = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "More than 8 hours": 4,
}

# Apply the mapping to the column and replace all other values with "Other"
df_train['Sleep Duration'] = df_train['Sleep Duration'].map(sleep_mapping).fillna(0)
df_test['Sleep Duration'] = df_test['Sleep Duration'].map(sleep_mapping).fillna(0)

for column in df_train:
    num_unique = df_train[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")

# Profession column pre processing 1- 37 rows and 1- 30 in city:
# Combine both datasets to calculate the frequency of each profession
combined_data_profession = pd.concat([df_train['Profession'], df_test['Profession']])

# Calculate value counts
profession_counts = combined_data_profession.value_counts()
rare_professions = profession_counts[profession_counts < 100].index

df_train['Profession'] = df_train['Profession'].fillna("Other").apply(lambda x: 'Other' if x in rare_professions else x)
df_test['Profession'] = df_test['Profession'].fillna("Other").apply(lambda x: 'Other' if x in rare_professions else x)

# City
combined_data_city = pd.concat([df_train['City'], df_test['City']])
city_count = combined_data_city.value_counts()
rare_cities = city_count[city_count < 100].index

df_train['City'] = df_train['City'].fillna('Other').apply(lambda x: 'Other' if x in rare_cities else x)
df_test['City'] = df_test['City'].fillna('Other').apply(lambda x: 'Other' if x in rare_cities else x)

# degree
combined_data_degree = pd.concat([df_train['Degree'], df_test['Degree']])
degree_count = combined_data_degree.value_counts()
degree_count.to_csv('degree_count.csv')
rare_degrees = degree_count[degree_count < 100].index

df_train['Degree'] = df_train['Degree'].fillna('Other').apply(lambda x: 'Other' if x in rare_degrees else x)
df_test['Degree'] = df_test['Degree'].fillna('Other').apply(lambda x: 'Other' if x in rare_degrees else x)

# Categorical data not encoded City, Degree and Name
df_train.drop(['Name'], axis=1, inplace=True)
df_test.drop(['Name'], axis=1, inplace=True)

non_numeric_cols = df_train.select_dtypes(include='object').columns
print('Non numeric columns:', non_numeric_cols)

label_encoder = LabelEncoder()
for col in non_numeric_cols:
    label_encoder.fit(df_train[col])  # Fit on training data
    df_train[col] = label_encoder.transform(df_train[col])
    df_test[col] = label_encoder.transform(df_test[col])


X_train = df_train.drop(['id', 'Depression'], axis=1)
y_train = df_train['Depression']

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                    stratify=y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Ensure the test data has the same feature set as the training data
df_test_features = df_test.drop(['id'], axis=1)

# Make predictions
test_predictions = rf.predict(df_test_features)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': df_test['id'],  # Include the 'id' column from the test dataset
    'Depression': test_predictions  # Predicted classes
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print(submission.head)