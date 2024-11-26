import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Display basic information about the training data
print("First rows of train csv file:")
print(df_train.head())
print("\nColumn names and data type of each column:")
print(df_train.dtypes)
print("\nCheck missing values in each column:")
print(df_train.isnull().sum())
print("\nShape of data:")
print(df_test.shape, '\n')

# Combine 'Work Pressure' and 'Academic Pressure' into a single 'Pressure' column
df_train['Pressure'] = df_train['Work Pressure'].fillna(df_train['Academic Pressure'])
df_test['Pressure'] = df_test['Work Pressure'].fillna(df_test['Academic Pressure'])
df_train.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)
df_test.drop(['Work Pressure', 'Academic Pressure'], axis=1, inplace=True)

# Combine 'Study Satisfaction' and 'Job Satisfaction' into a single 'Satisfaction' column
df_train['Satisfaction'] = df_train['Study Satisfaction'].fillna(df_train['Job Satisfaction'])
df_test['Satisfaction'] = df_test['Study Satisfaction'].fillna(df_test['Job Satisfaction'])
df_train.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)
df_test.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

# Merge 'Working Professional or Student' into 'Profession'
df_train['Profession'] = np.where(df_train['Working Professional or Student'] == 'Student', 'Student',
                                  df_train['Profession'])
df_test['Profession'] = np.where(df_test['Working Professional or Student'] == 'Student', 'Student',
                                 df_test['Profession'])

# Drop the 'Working Professional or Student' column
df_train.drop(['Working Professional or Student'], axis=1, inplace=True)
df_test.drop(['Working Professional or Student'], axis=1, inplace=True)

# Fill missing values in 'Profession' with 'Other'
df_train['Profession'] = df_train['Profession'].fillna("Other")
df_test['Profession'] = df_test['Profession'].fillna("Other")

# Map binary columns (e.g., 'Yes'/'No', 'Male'/'Female') to numeric values
binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
df_train['Gender'] = df_train['Gender'].map(binary_mapping)
df_test['Gender'] = df_test['Gender'].map(binary_mapping)

df_train['Have you ever had suicidal thoughts ?'] = df_train['Have you ever had suicidal thoughts ?'].map(
    binary_mapping)
df_test['Have you ever had suicidal thoughts ?'] = df_test['Have you ever had suicidal thoughts ?'].map(binary_mapping)

df_train['Family History of Mental Illness'] = df_train['Family History of Mental Illness'].map(binary_mapping)
df_test['Family History of Mental Illness'] = df_test['Family History of Mental Illness'].map(binary_mapping)

# Map 'Dietary Habits' to numeric values
dietary_mapping = {'Moderate': 1, 'Unhealthy': 2, 'Healthy': 3}
df_train['Dietary Habits'] = df_train['Dietary Habits'].map(dietary_mapping).fillna(0)
df_test['Dietary Habits'] = df_test['Dietary Habits'].map(dietary_mapping).fillna(0)

# Map 'Sleep Duration' categories to numeric values
sleep_mapping = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4}
df_train['Sleep Duration'] = df_train['Sleep Duration'].map(sleep_mapping).fillna(0)
df_test['Sleep Duration'] = df_test['Sleep Duration'].map(sleep_mapping).fillna(0)

# Display the number of unique categories in each column for insight
for column in df_train:
    num_unique = df_train[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")

# Process rare professions
combined_data_profession = pd.concat([df_train['Profession'], df_test['Profession']])
profession_counts = combined_data_profession.value_counts()
rare_professions = profession_counts[profession_counts < 100].index
df_train['Profession'] = df_train['Profession'].apply(lambda x: 'Other' if x in rare_professions else x)
df_test['Profession'] = df_test['Profession'].apply(lambda x: 'Other' if x in rare_professions else x)

# Process rare cities
combined_data_city = pd.concat([df_train['City'], df_test['City']])
city_count = combined_data_city.value_counts()
rare_cities = city_count[city_count < 100].index
df_train['City'] = df_train['City'].apply(lambda x: 'Other' if x in rare_cities else x)
df_test['City'] = df_test['City'].apply(lambda x: 'Other' if x in rare_cities else x)

# Process rare degrees
combined_data_degree = pd.concat([df_train['Degree'], df_test['Degree']])
degree_count = combined_data_degree.value_counts()
rare_degrees = degree_count[degree_count < 100].index
df_train['Degree'] = df_train['Degree'].apply(lambda x: 'Other' if x in rare_degrees else x)
df_test['Degree'] = df_test['Degree'].apply(lambda x: 'Other' if x in rare_degrees else x)

# Drop the 'Name' column as it's not needed for modeling
df_train.drop(['Name'], axis=1, inplace=True)
df_test.drop(['Name'], axis=1, inplace=True)

df_train.drop(['CGPA'], axis=1, inplace=True)
df_test.drop(['CGPA'], axis=1, inplace=True)

# Encode remaining non-numeric columns using LabelEncoder
non_numeric_cols = df_train.select_dtypes(include='object').columns
numeric_cols = df_train.select_dtypes(include=['float64']).columns
print('\nNon numeric columns:', non_numeric_cols)

# Replace NaN in numeric columns with a default float value
df_train[numeric_cols] = df_train[numeric_cols].fillna(0.0)
df_test[numeric_cols] = df_test[numeric_cols].fillna(0.0)

# Replace NaN with 'Unknown' in non-numeric columns
df_train[non_numeric_cols] = df_train[non_numeric_cols].fillna('Unknown')
df_test[non_numeric_cols] = df_test[non_numeric_cols].fillna('Unknown')

# Apply LabelEncoder to non-numeric columns
label_encoder = LabelEncoder()
for col in non_numeric_cols:
    df_train[col] = label_encoder.fit_transform(df_train[col])
    df_test[col] = label_encoder.transform(df_test[col])

# Split the data into features (X) and target variable (y)
X_train = df_train.drop(['id', 'Depression'], axis=1)
y_train = df_train['Depression']

# Split training data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Calculate and display evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Prepare test data and make predictions for submission
df_test_features = df_test.drop(['id'], axis=1)
test_predictions = rf.predict(df_test_features)

# Create and save the submission file
submission = pd.DataFrame({'id': df_test['id'], 'Depression': test_predictions})
submission.to_csv('submission.csv', index=False)
print(submission.head())
