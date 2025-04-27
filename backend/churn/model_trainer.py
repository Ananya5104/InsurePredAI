import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import joblib
import pickle

def train_models():
    """
    Train all necessary models using the existing dataset.
    This function will train:
    1. Churn prediction model (LinearSVC)
    2. Plan type recommender for non-churning customers (LinearSVC)
    3. Plan type recommender for churning customers (LinearSVC)
    4. XGBoost model for interactive recommendations (ml_model.pkl)

    Returns:
        dict: Dictionary containing all trained models and scalers
    """
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    LOGS_DIR = os.path.join(PARENT_DIR, "logs")
    MODELS_DIR = os.path.join(PARENT_DIR, "models")

    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Define training data path
    training_data_path = os.path.join(LOGS_DIR, "training_data.csv")

    # Load the dataset
    print(f"Loading training data from {training_data_path}")
    df = pd.read_csv(training_data_path)

    # Define columns to scale
    cols_to_scale = ['Age', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)']
    scaling_cols = ['Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)', 'days_passed']

    # Train churn prediction model
    print("Training churn prediction model...")
    churn_model, churn_scaler = train_churn_model(df, cols_to_scale)

    # Create separate dataframes for churning and non-churning customers
    print("Preparing data for plan recommender models...")
    churning_df = df[df.iloc[:, -1] > 0.5].copy()  # Assuming last column is churn
    non_churning_df = df[df.iloc[:, -1] <= 0.5].copy()

    # Train plan recommender models
    print("Training plan recommender for non-churning customers...")
    plan_model, plan_scaler = train_plan_recommender(non_churning_df, cols_to_scale)

    print("Training plan recommender for churning customers...")
    plan_model_churn, plan_scaler_churn = train_plan_recommender(churning_df, cols_to_scale)

    # Train XGBoost model (ml_model.pkl) from final_insurance_recommender notebook
    print("Training XGBoost model for interactive recommendations...")
    xgb_model, xgb_scaler = train_xgboost_model(df, scaling_cols)

    # Save models and scalers
    print("Saving models and scalers...")
    save_models(MODELS_DIR, churn_model, churn_scaler, plan_model, plan_scaler,
                plan_model_churn, plan_scaler_churn, xgb_model, xgb_scaler)

    # Return models and scalers
    return {
        'churn_model': churn_model,
        'churn_scaler': churn_scaler,
        'plan_model': plan_model,
        'plan_scaler': plan_scaler,
        'plan_model_churn': plan_model_churn,
        'plan_scaler_churn': plan_scaler_churn,
        'xgb_model': xgb_model,
        'xgb_scaler': xgb_scaler
    }

def train_churn_model(df, cols_to_scale):
    """Train the churn prediction model using LinearSVC as in the notebook"""
    # Prepare data
    y = df.iloc[:, -1]  # Last column is churn
    X = df.iloc[:, :-1]  # All columns except the last one

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale features
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Train model - using LinearSVC as in the notebook
    model = LinearSVC(dual='auto', random_state=101)
    model.fit(X_train, y_train)

    return model, scaler

def train_plan_recommender(df, cols_to_scale):
    """Train a plan type recommender model using LinearSVC as in the notebook"""
    # Check if dataframe is empty or too small
    if len(df) < 10:
        print("Not enough data for plan recommender. Creating a dummy model...")
        # Return a dummy model that always recommends plan type 2 (Standard)
        class DummyModel:
            def predict(self, X):
                return np.array([2] * len(X))

            def decision_function(self, X):
                return np.array([0] * len(X))

        return DummyModel(), None

    # Prepare data - assuming Plan Type is in the dataset
    if 'Plan Type' in df.columns:
        X = df.drop(['Plan Type', 'Churn'], axis=1, errors='ignore')
        y = df['Plan Type']
    else:
        # If Plan Type is not in the dataset, use a dummy model
        print("Plan Type column not found. Creating a dummy model...")
        class DummyModel:
            def predict(self, X):
                return np.array([2] * len(X))

            def decision_function(self, X):
                return np.array([0] * len(X))

        return DummyModel(), None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale features
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Train model - using LinearSVC as in the notebook
    model = LinearSVC(dual='auto', random_state=101)
    model.fit(X_train, y_train)

    return model, scaler

def train_xgboost_model(df, scaling_cols):
    """Train the XGBoost model as used in the final_insurance_recommender notebook"""
    # Prepare data
    y = df['Churn']
    X = df.drop("Churn", axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale features
    scaler = StandardScaler()
    if all(col in X_train.columns for col in scaling_cols):
        X_train[scaling_cols] = scaler.fit_transform(X_train[scaling_cols])
        X_test[scaling_cols] = scaler.transform(X_test[scaling_cols])
    else:
        # If scaling columns are not in the dataset, use available numeric columns
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Train XGBoost model with parameters from the notebook
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    return model, scaler

def save_models(models_dir, churn_model, churn_scaler, plan_model, plan_scaler,
                plan_model_churn, plan_scaler_churn, xgb_model, xgb_scaler):
    """Save all models and scalers to disk"""
    # Save churn model and scaler
    joblib.dump(churn_model, os.path.join(models_dir, "churn_model.pkl"))
    joblib.dump(churn_scaler, os.path.join(models_dir, "churn_scaler.pkl"))

    # Save plan recommender models and scalers
    joblib.dump(plan_model, os.path.join(models_dir, "plan_type_recommender.pkl"))
    joblib.dump(plan_scaler, os.path.join(models_dir, "plan_type_scaler.pkl"))

    joblib.dump(plan_model_churn, os.path.join(models_dir, "plan_type_recommender_churn.pkl"))
    joblib.dump(plan_scaler_churn, os.path.join(models_dir, "plan_type_scaler_churn.pkl"))

    # Save XGBoost model (ml_model.pkl) using pickle as in the notebook
    with open(os.path.join(models_dir, "ml_model.pkl"), "wb") as file:
        pickle.dump(xgb_model, file)

    # Also save the XGBoost scaler
    joblib.dump(xgb_scaler, os.path.join(models_dir, "ml_model_scaler.pkl"))

    print(f"All models and scalers saved to {models_dir}")

if __name__ == "__main__":
    # Test the function
    train_models()
