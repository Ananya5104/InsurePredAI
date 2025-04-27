import requests
import json

# Define the API endpoint
url = "http://localhost:8000/api/predict/"

# Define the input data
data = {
    "features": [
        35,  # Age
        1,   # Gender (1 for Male, 0 for Female)
        60000,  # Earnings ($)
        500,    # Claim Amount ($)
        8000,   # Insurance Plan Amount ($)
        1,      # Credit Score (1 for Good, 0 for Bad)
        1,      # Marital Status (1 for Married, 0 for Single)
        1000,   # days_passed
        0,      # Automobile Insurance
        1,      # Health Insurance
        0,      # Life Insurance
        2       # Plan Type (1 for Basic, 2 for Standard, 3 for Premium)
    ],
    "raw_data": {
        "age": 35,
        "gender": "M",
        "earnings": 60000,
        "claim_amount": 500,
        "insurance_plan_amount": 8000,
        "credit_score": True,
        "marital_status": "M",
        "days_passed": 1000,
        "type_of_insurance": "health",
        "plan_type": "premium"
    }
}

# Convert data to JSON
json_data = json.dumps(data)

# Set headers
headers = {
    "Content-Type": "application/json"
}

# Make the API call
try:
    response = requests.post(url, data=json_data, headers=headers)
    
    # Print the response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {str(e)}")
