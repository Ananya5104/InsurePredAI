import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Path to the training data
training_data_path = 'backend/logs/training_data.csv'

# Define column names
column_names = [
    'Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)', 
    'Credit Score', 'Marital Status', 'days_passed', 'Automobile Insurance', 
    'Health Insurance', 'Life Insurance', 'Plan Type', 'Churn'
]

# Check if the file has headers by reading the first line
with open(training_data_path, 'r') as f:
    first_line = f.readline().strip()

# If the first line contains only numbers and commas, it's likely there are no headers
if all(c.isdigit() or c in ',.+-' for c in first_line.replace(',', '')):
    print("Dataset appears to have no headers. Adding column names...")
    # Read CSV with specified column names
    df = pd.read_csv(training_data_path, header=None, names=column_names)
else:
    # Read CSV normally
    df = pd.read_csv(training_data_path)

# Extract features and target
X = df.iloc[:,:12]
y = df.iloc[:,12]

print("Training a simple churn prediction model...")
# Define columns to scale
cols_to_scale = ['Age', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)']

# Train a simple model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Scale features
scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Train model
model = LinearSVC(dual='auto', random_state=101)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model and scaler
os.makedirs('backend/models', exist_ok=True)
joblib.dump(model, 'backend/models/churn_model.pkl')
joblib.dump(scaler, 'backend/models/churn_scaler.pkl')

print("Model saved successfully!")

# Now let's test the churn prediction
print("\nTesting churn prediction...")

# Create a sample customer
sample_customer = {
    'Age': 35,
    'Gender': 1,
    'Earnings ($)': 60000,
    'Claim Amount ($)': 500,
    'Insurance Plan Amount ($)': 8000,
    'Credit Score': 1,
    'Marital Status': 1,
    'days_passed': 1000,
    'Automobile Insurance': 0,
    'Health Insurance': 1,
    'Life Insurance': 0,
    'Plan Type': 2
}

# Convert to DataFrame
input_data = pd.DataFrame([sample_customer])

# Scale the features
input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

# Make prediction
try:
    # Try using predict_proba
    if hasattr(model, 'predict_proba'):
        churn_probability = model.predict_proba(input_data)[0][1]
        print(f"Churn probability (using predict_proba): {churn_probability:.4f}")
    else:
        # For LinearSVC, use decision_function
        decision_value = model.decision_function(input_data)[0]
        # Convert to probability using sigmoid function
        churn_probability = 1 / (1 + np.exp(-decision_value))
        print(f"Churn probability (using decision_function): {churn_probability:.4f}")
    
    # Also try direct prediction
    prediction = model.predict(input_data)[0]
    print(f"Direct prediction: {prediction} ({'Churn' if prediction > 0.5 else 'No Churn'})")
    
except Exception as e:
    print(f"Error making prediction: {str(e)}")

print("\nDone!")
