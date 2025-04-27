import joblib
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Import the model trainer
from .model_trainer import train_models

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
PARENT_DIR = os.path.dirname(BASE_DIR)  # Moves one level up
MODELS_DIR = os.path.join(PARENT_DIR, "models")

# Check if models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

# Check if models exist, if not train them
model_files = [
    os.path.join(MODELS_DIR, "churn_model.pkl"),
    os.path.join(MODELS_DIR, "plan_type_recommender.pkl"),
    os.path.join(MODELS_DIR, "plan_type_recommender_churn.pkl")
]

# Train or load models
try:
    if not all(os.path.exists(file) for file in model_files):
        print("Training models...")
        models_dict = train_models()
        churn_model = models_dict['churn_model']
        churn_scaler = models_dict['churn_scaler']
        plan_recommender = models_dict['plan_model']
        plan_scaler = models_dict['plan_scaler']
        plan_recommender_churn = models_dict['plan_model_churn']
        plan_scaler_churn = models_dict['plan_scaler_churn']

        # Also get the XGBoost model if available
        if 'xgb_model' in models_dict:
            xgb_model = models_dict['xgb_model']
            xgb_scaler = models_dict['xgb_scaler']
    else:
        # Load all models
        try:
            churn_model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
            churn_model = joblib.load(churn_model_path)

            plan_recommender_path = os.path.join(MODELS_DIR, "plan_type_recommender.pkl")
            plan_recommender = joblib.load(plan_recommender_path)

            plan_recommender_churn_path = os.path.join(MODELS_DIR, "plan_type_recommender_churn.pkl")
            plan_recommender_churn = joblib.load(plan_recommender_churn_path)

            # Load scalers
            churn_scaler_path = os.path.join(MODELS_DIR, "churn_scaler.pkl")
            plan_scaler_path = os.path.join(MODELS_DIR, "plan_type_scaler.pkl")
            plan_scaler_churn_path = os.path.join(MODELS_DIR, "plan_type_scaler_churn.pkl")

            churn_scaler = joblib.load(churn_scaler_path)
            plan_scaler = joblib.load(plan_scaler_path)
            plan_scaler_churn = joblib.load(plan_scaler_churn_path)

            # Try to load XGBoost model if it exists
            xgb_model_path = os.path.join(MODELS_DIR, "ml_model.pkl")
            xgb_scaler_path = os.path.join(MODELS_DIR, "ml_model_scaler.pkl")

            if os.path.exists(xgb_model_path):
                with open(xgb_model_path, 'rb') as f:
                    xgb_model = pickle.load(f)

                if os.path.exists(xgb_scaler_path):
                    xgb_scaler = joblib.load(xgb_scaler_path)
                else:
                    xgb_scaler = None
        except Exception as e:
            print(f"Error loading models: {str(e)}. Training new models...")
            models_dict = train_models()
            churn_model = models_dict['churn_model']
            churn_scaler = models_dict['churn_scaler']
            plan_recommender = models_dict['plan_model']
            plan_scaler = models_dict['plan_scaler']
            plan_recommender_churn = models_dict['plan_model_churn']
            plan_scaler_churn = models_dict['plan_scaler_churn']

            # Also get the XGBoost model if available
            if 'xgb_model' in models_dict:
                xgb_model = models_dict['xgb_model']
                xgb_scaler = models_dict['xgb_scaler']
except Exception as e:
    print(f"Critical error in model loading/training: {str(e)}. Using dummy models...")

    # Create dummy models as a last resort
    from sklearn.dummy import DummyClassifier
    churn_model = DummyClassifier(strategy="constant", constant=0)
    churn_model.fit(np.array([[0, 0]]), np.array([0, 0]))
    churn_scaler = None

    class DummyModel:
        def predict(self, X):
            return np.array([2] * len(X))

        def decision_function(self, X):
            return np.array([0] * len(X))

        def predict_proba(self, X):
            return np.array([[0, 0, 1]] * len(X))

    plan_recommender = DummyModel()
    plan_scaler = None
    plan_recommender_churn = DummyModel()
    plan_scaler_churn = None

# Define the feature names as per training data
feature_names = ['Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)',
                'Insurance Plan Amount ($)', 'Credit Score', 'Marital Status', 'days_passed',
                'Automobile Insurance', 'Health Insurance', 'Life Insurance', 'Plan Type']



# Use the correct path to the training data file
LOGS_DIR = os.path.join(PARENT_DIR, "logs")
training_data_path = os.path.join(LOGS_DIR, "training_data.csv")

# Check if training data exists
if not os.path.exists(training_data_path):
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    # Create a sample dataset using the function from model_trainer
    from .model_trainer import create_sample_dataset
    create_sample_dataset(training_data_path)

# Load the training data with proper column names
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
    customer_profiles = pd.read_csv(training_data_path, header=None, names=column_names)
else:
    # Read CSV normally
    customer_profiles = pd.read_csv(training_data_path)

    # If the dataset has numeric column names, rename them
    if not any(col in customer_profiles.columns for col in column_names):
        print("Dataset has numeric column names. Renaming columns...")
        # Map numeric columns to expected names based on position
        if len(customer_profiles.columns) >= 13:  # Assuming at least 13 columns
            new_columns = {
                customer_profiles.columns[0]: 'Age',
                customer_profiles.columns[1]: 'Gender',
                customer_profiles.columns[2]: 'Earnings ($)',
                customer_profiles.columns[3]: 'Claim Amount ($)',
                customer_profiles.columns[4]: 'Insurance Plan Amount ($)',
                customer_profiles.columns[5]: 'Credit Score',
                customer_profiles.columns[6]: 'Marital Status',
                customer_profiles.columns[7]: 'days_passed',
                customer_profiles.columns[8]: 'Automobile Insurance',
                customer_profiles.columns[9]: 'Health Insurance',
                customer_profiles.columns[10]: 'Life Insurance',
                customer_profiles.columns[11]: 'Plan Type',
                customer_profiles.columns[12]: 'Churn'
            }
            customer_profiles = customer_profiles.rename(columns=new_columns)

# Extract features and target
X = customer_profiles.iloc[:,:12]
y = customer_profiles.iloc[:,12]

def get_similar_customers(target_customer, n=10):
    """
    Find similar customers based on cosine similarity
    """
    # Convert target_customer to a 2D numpy array with shape (1, n_features)
    if isinstance(target_customer, pd.Series):
        target_array = target_customer.values.reshape(1, -1)
    else:
        target_array = target_customer.values.reshape(1, -1)

    # Convert X to a 2D numpy array
    X_array = X.values
    # Calculate similarity directly with numpy arrays
    similarity_matrix = cosine_similarity(target_array, X_array)

    # Get indices of most similar customers
    similar_indices = similarity_matrix[0].argsort()[::-1][:n]

    return similar_indices
# Function to generate personalized recommendations based on similar customers
def generate_recommendations(target_customer):
    """
    Generate personalized recommendations based on similar non-churned customers

    Parameters:
    target_customer (pd.Series): The customer profile to generate recommendations for

    Returns:
    dict: Dictionary of recommendations
    """

    # Get similar customers
    similar_indices = get_similar_customers(target_customer)

    # Ensure target_customer is a Series (not a DataFrame)
    if hasattr(target_customer, 'iloc') and not isinstance(target_customer, pd.Series):
        target_customer = target_customer.iloc[0]
    # Filter to non-churned similar customers
    non_churned_similar = [idx for idx in similar_indices if y.iloc[idx] == 0]

    if not non_churned_similar:
        return {"General": ["We don't have enough similar customers to provide personalized recommendations."]}

    similar_customers_data = X.iloc[non_churned_similar]

    recommendations = {}


    # 1. Insurance Types comparison
    # Check which insurance types are common among similar non-churned customers
    auto_insurance_popular = similar_customers_data['Automobile Insurance'].mean() > 0.5
    health_insurance_popular = similar_customers_data['Health Insurance'].mean() > 0.5
    life_insurance_popular = similar_customers_data['Life Insurance'].mean() > 0.5

    insurance_recs = []
    if auto_insurance_popular and target_customer['Automobile Insurance'] == 0:
        insurance_recs.append("Add automobile insurance - popular among similar customers who stay with us")
    if health_insurance_popular and target_customer['Health Insurance'] == 0:
        insurance_recs.append("Include health insurance coverage - common among customers with your profile")
    if life_insurance_popular and target_customer['Life Insurance'] == 0:
        insurance_recs.append("Consider life insurance protection - beneficial for customers similar to you")

    if insurance_recs:
        recommendations['Insurance_Options'] = insurance_recs
        print(recommendations["Insurance_Options"])  # Only print if the key exists
    # 2. Analyze claim behaviors of similar customers
    avg_claim = similar_customers_data['Claim Amount ($)'].mean()
    if target_customer['Claim Amount ($)'] < avg_claim - 0.5:
        recommendations['Claim_Optimization'] = [
            "You may be under-utilizing your benefits compared to similar customers",
            "Schedule a coverage review to ensure you're getting the most from your plan"
        ]
    elif target_customer['Claim Amount ($)'] > avg_claim + 0.5:
        recommendations['Claim_Optimization'] = [
            "Your claim pattern differs from similar satisfied customers",
            "Consider our premium protection plan with higher claim limits"
        ]

    # 3. Credit score-based recommendations
    avg_credit = similar_customers_data['Credit Score'].mean()
    if target_customer['Credit Score'] < avg_credit - 0.1:
        recommendations['Credit_Improvement'] = [
            "Our credit improvement program can help enhance your insurance terms",
            "Customers with improved credit scores often receive better rates"
        ]

    # 4. Additional demographic insights
    age = target_customer['Age']
    if age < 30:
        recommendations['Young_Customer'] = [
            "Short-term flexible coverage plans for young professionals",
            "Digital service with mobile app benefits"
        ]
    elif age > 55:
        recommendations['Senior_Customer'] = [
            "Fixed premium rates for long-term loyalty",
            "Priority human customer support"
        ]

    # If no specific recommendations were generated, provide a general one
    if not recommendations:
        recommendations['General'] = ["Based on similar customers, your current plan appears optimal."]

    return recommendations
def get_comprehensive_analysis(features):
    """
    Combined function that performs churn prediction, plan recommendation,
    and customer value analysis in one call.

    Parameters:
    - features: List of customer features in the same order as feature_names

    Returns:
    - Dictionary with all analysis results
    """
    # Create DataFrame with all features
    input_data = pd.DataFrame([features], columns=feature_names).astype(float)
    temp=input_data.copy()
    # Define the columns to be scaled
    scale_columns = ['Age', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)']

    # 1. CHURN PREDICTION
    # Create a copy for churn prediction to avoid affecting the original
    churn_input = input_data.copy()

    # Scale only the selected columns for churn prediction
    if churn_scaler is not None:
        churn_input.loc[:, scale_columns] = churn_scaler.transform(churn_input[scale_columns])

    # Get churn probability
    try:
        # Check if the model has predict_proba method
        if hasattr(churn_model, 'predict_proba'):
            churn_probability = churn_model.predict_proba(churn_input)[0][1]
        else:
            # For LinearSVC, we'll use the decision function which gives the distance to the hyperplane
            # and convert it to a probability-like value between 0 and 1 using sigmoid function
            decision_value = churn_model.decision_function(churn_input)[0]
            # Convert to a probability-like value using sigmoid function
            churn_probability = 1 / (1 + np.exp(-decision_value))
    except Exception as e:
        print(f"Error getting churn probability: {str(e)}. Using default value.")
        # Default to a moderate churn probability
        churn_probability = 0.3

    is_churn_risk = churn_probability > 0.5

    churn_result = {
        "churn_probability": float(churn_probability),
        "is_churn_risk": is_churn_risk,
        "recommendation": (
            "High churn risk! Consider offering personalized discounts or improved benefits."
            if is_churn_risk else
            "Low churn risk. Maintain regular engagement and customer satisfaction measures."
        )
    }

    # 2. PLAN RECOMMENDATION
    # Create a copy for plan recommendation
    plan_input = input_data.copy()

    # Apply appropriate scaling based on churn risk
    if is_churn_risk and plan_scaler_churn is not None:
        plan_input.loc[:, scale_columns] = plan_scaler_churn.transform(plan_input[scale_columns])
        recommended_plan = plan_recommender_churn.predict(plan_input.loc[:, plan_input.columns != "Plan Type"])[0]
    elif plan_scaler is not None:
        plan_input.loc[:, scale_columns] = plan_scaler.transform(plan_input[scale_columns])
        recommended_plan = plan_recommender.predict(plan_input.loc[:, plan_input.columns != "Plan Type"])[0]
    else:
        # Handle case where scalers are None
        recommended_plan = features[11]  # Default to current plan

    # Convert numeric plan type to descriptive name
    plan_names = {1: "Basic", 2: "Standard", 3: "Premium"}

    # Handle potential errors with current plan
    try:
        current_plan_num = int(float(features[11]))
        current_plan = plan_names.get(current_plan_num, "Standard")  # Default to Standard if not found
    except (ValueError, IndexError, TypeError):
        # If there's any error, default to Standard
        current_plan = "Standard"

    # Handle potential errors with recommended plan
    try:
        recommended_plan_num = int(float(recommended_plan))
        recommended_plan_name = plan_names.get(recommended_plan_num, "Standard")
    except (ValueError, TypeError):
        recommended_plan_name = "Standard"

    # Generate recommendation message
    try:
        if int(float(recommended_plan)) == int(float(features[11])):  # Current plan is already optimal
            plan_message = f"The customer's current {current_plan} plan is already optimal based on their profile."
        else:
            plan_message = f"Recommend upgrading from {current_plan} to {recommended_plan_name} plan for better value and reduced churn risk."
    except (ValueError, TypeError):
        # If there's any error in comparison, provide a generic message
        plan_message = f"Recommend the {recommended_plan_name} plan based on the customer profile."

    # Ensure all values are valid before creating the result
    try:
        recommended_plan_int = int(float(recommended_plan))
    except (ValueError, TypeError):
        recommended_plan_int = 2  # Default to Standard (2)

    plan_result = {
        "recommended_plan": recommended_plan_int,
        "recommended_plan_name": recommended_plan_name or "Standard",
        "current_plan": current_plan or "Standard",
        "plan_message": plan_message or f"Recommend the {recommended_plan_name or 'Standard'} plan for optimal coverage."
    }

    # 3. CUSTOMER SIMILARITY ANALYSIS
    # Generate personalized recommendations based on similar customers
    try:
        # Get similar customers directly instead of calling generate_recommendations
        similar_indices = get_similar_customers(temp)

        # Filter to non-churned similar customers
        non_churned_similar = [idx for idx in similar_indices if y.iloc[idx] == 0]

        if not non_churned_similar:
            # Provide generic recommendations instead of saying we don't have enough data
            customer_recommendations = {
                "Insurance_Options": [
                    "Consider bundling multiple insurance types for better coverage and discounts",
                    "Our most popular insurance combinations include health and life insurance"
                ],
                "Plan_Optimization": [
                    f"The {current_plan} plan offers good value for your profile",
                    "Regular policy reviews can help ensure your coverage meets your changing needs"
                ],
                "Customer_Benefits": [
                    "Take advantage of our loyalty program for long-term customers",
                    "Our mobile app provides easy access to your policy information and claims"
                ]
            }
        else:
            similar_customers_data = X.iloc[non_churned_similar]

            # Create recommendations dictionary
            customer_recommendations = {}

            # 1. Insurance Types comparison
            auto_insurance_popular = similar_customers_data['Automobile Insurance'].mean() > 0.5
            health_insurance_popular = similar_customers_data['Health Insurance'].mean() > 0.5
            life_insurance_popular = similar_customers_data['Life Insurance'].mean() > 0.5

            insurance_recs = []
            if auto_insurance_popular and temp['Automobile Insurance'].iloc[0] == 0:
                insurance_recs.append("Add automobile insurance - popular among similar customers who stay with us")
            if health_insurance_popular and temp['Health Insurance'].iloc[0] == 0:
                insurance_recs.append("Include health insurance coverage - common among customers with your profile")
            if life_insurance_popular and temp['Life Insurance'].iloc[0] == 0:
                insurance_recs.append("Consider life insurance protection - beneficial for customers similar to you")

            if insurance_recs:
                customer_recommendations['Insurance_Options'] = insurance_recs

            # 2. Analyze claim behaviors of similar customers
            avg_claim = similar_customers_data['Claim Amount ($)'].mean()
            if temp['Claim Amount ($)'].iloc[0] < avg_claim - 0.5:
                customer_recommendations['Claim_Optimization'] = [
                    "You may be under-utilizing your benefits compared to similar customers",
                    "Schedule a coverage review to ensure you're getting the most from your plan"
                ]
            elif temp['Claim Amount ($)'].iloc[0] > avg_claim + 0.5:
                customer_recommendations['Claim_Optimization'] = [
                    "Your claim pattern differs from similar satisfied customers",
                    "Consider our premium protection plan with higher claim limits"
                ]

            # 3. Credit score-based recommendations
            avg_credit = similar_customers_data['Credit Score'].mean()
            if temp['Credit Score'].iloc[0] < avg_credit - 0.1:
                customer_recommendations['Credit_Improvement'] = [
                    "Our credit improvement program can help enhance your insurance terms",
                    "Customers with improved credit scores often receive better rates"
                ]

            # 4. Additional demographic insights
            age = temp['Age'].iloc[0]
            if age < 30:
                customer_recommendations['Young_Customer'] = [
                    "Short-term flexible coverage plans for young professionals",
                    "Digital service with mobile app benefits"
                ]
            elif age > 55:
                customer_recommendations['Senior_Customer'] = [
                    "Fixed premium rates for long-term loyalty",
                    "Priority human customer support"
                ]

            # If no specific recommendations were generated, provide a general one
            if not customer_recommendations:
                customer_recommendations['General'] = ["Based on similar customers, your current plan appears optimal."]
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        customer_recommendations = {"General": ["Unable to generate personalized recommendations."]}

    # Return combined results
    return {
        "churn_analysis": churn_result,
        "plan_recommendation": plan_result,
        "customer_recommendations": customer_recommendations
    }