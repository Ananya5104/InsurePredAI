import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os

# Path to the training data
training_data_path = 'backend/logs/training_data.csv'

# Check if the file has headers by reading the first line
with open(training_data_path, 'r') as f:
    first_line = f.readline().strip()

# Define column names
column_names = [
    'Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)', 
    'Credit Score', 'Marital Status', 'days_passed', 'Automobile Insurance', 
    'Health Insurance', 'Life Insurance', 'Plan Type', 'Churn'
]

# If the first line contains only numbers and commas, it's likely there are no headers
if all(c.isdigit() or c in ',.+-' for c in first_line.replace(',', '')):
    print("Dataset appears to have no headers. Adding column names...")
    # Read CSV with specified column names
    df = pd.read_csv(training_data_path, header=None, names=column_names)
else:
    # Read CSV normally
    df = pd.read_csv(training_data_path)

# Define columns to scale
cols_to_scale = ['Age', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)']

# Train churn prediction model
print("Training churn prediction model...")

# Prepare data
if 'Churn' in df.columns:
    y = df['Churn']
    X = df.drop('Churn', axis=1)
else:
    # Assume last column is churn
    y = df.iloc[:, -1]  # Last column is churn
    X = df.iloc[:, :-1]  # All columns except the last one

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Scale features
scaler = StandardScaler()
try:
    # Try to scale the specified columns
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print("Successfully scaled the data!")
except KeyError as e:
    print(f"Warning: Could not scale specified columns: {e}")
    # Fall back to scaling all numeric columns
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        print(f"Scaled numeric columns instead: {numeric_cols}")

# Train model - using LinearSVC as in the notebook
try:
    model = LinearSVC(dual='auto', random_state=101)
    model.fit(X_train, y_train)
    print("Successfully trained the LinearSVC model!")
    
    # Test the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"Error training LinearSVC: {e}")

print("Model training completed!")
