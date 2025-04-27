import pandas as pd

# Path to the training data
training_data_path = 'backend/logs/training_data.csv'

# Check if the file has headers by reading the first line
with open(training_data_path, 'r') as f:
    first_line = f.readline().strip()

print("First line of the CSV file:", first_line)

# If the first line contains only numbers and commas, it's likely there are no headers
if all(c.isdigit() or c in ',.+-' for c in first_line.replace(',', '')):
    print("Dataset appears to have no headers. Adding column names...")
    # Define column names
    column_names = [
        'Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)', 
        'Credit Score', 'Marital Status', 'days_passed', 'Automobile Insurance', 
        'Health Insurance', 'Life Insurance', 'Plan Type', 'Churn'
    ]
    
    # Read CSV with specified column names
    df = pd.read_csv(training_data_path, header=None, names=column_names)
else:
    # Read CSV normally
    df = pd.read_csv(training_data_path)

print("DataFrame columns:", df.columns.tolist())
print("DataFrame shape:", df.shape)
print("First 5 rows:")
print(df.head(5))
