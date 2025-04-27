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

    # Check if the file has headers by reading the first line
    with open(training_data_path, 'r') as f:
        first_line = f.readline().strip()

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

        # Check if the dataset has the expected column names
        expected_cols = ['Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)',
                        'Credit Score', 'Marital Status', 'days_passed', 'Automobile Insurance',
                        'Health Insurance', 'Life Insurance', 'Plan Type', 'Churn']

        # If the dataset has numeric column names, rename them
        if not any(col in df.columns for col in expected_cols):
            print("Dataset has numeric column names. Renaming columns...")
            # Map numeric columns to expected names based on position
            if len(df.columns) >= 13:  # Assuming at least 13 columns
                new_columns = {
                    df.columns[0]: 'Age',
                    df.columns[1]: 'Gender',
                    df.columns[2]: 'Earnings ($)',
                    df.columns[3]: 'Claim Amount ($)',
                    df.columns[4]: 'Insurance Plan Amount ($)',
                    df.columns[5]: 'Credit Score',
                    df.columns[6]: 'Marital Status',
                    df.columns[7]: 'days_passed',
                    df.columns[8]: 'Automobile Insurance',
                    df.columns[9]: 'Health Insurance',
                    df.columns[10]: 'Life Insurance',
                    df.columns[11]: 'Plan Type',
                    df.columns[12]: 'Churn'
                }
                df = df.rename(columns=new_columns)

    # Define columns to scale - now using the renamed columns
    cols_to_scale = ['Age', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)']
    scaling_cols = ['Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)', 'days_passed']

    # Check if all scaling columns exist in the dataframe
    missing_cols = [col for col in cols_to_scale if col not in df.columns]
    if missing_cols:
        print(f"Warning: Some scaling columns are missing: {missing_cols}")
        # Use only available columns for scaling
        cols_to_scale = [col for col in cols_to_scale if col in df.columns]
        scaling_cols = [col for col in scaling_cols if col in df.columns]

        # If no scaling columns are available, use numeric columns
        if not cols_to_scale:
            print("Using numeric columns for scaling instead")
            cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:4]  # Use first 4 numeric columns
            scaling_cols = cols_to_scale

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
    except KeyError as e:
        print(f"Warning: Could not scale specified columns: {e}")
        # Fall back to scaling all numeric columns
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Train model - using LinearSVC as in the notebook
    try:
        model = LinearSVC(dual='auto', random_state=101)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training LinearSVC: {e}. Using XGBClassifier as fallback.")
        # Fallback to XGBClassifier if LinearSVC fails
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
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

            def predict_proba(self, X):
                return np.array([[0, 0, 1]] * len(X))  # Always predict class 2 with 100% probability

        return DummyModel(), None

    # Prepare data - assuming Plan Type is in the dataset
    if 'Plan Type' in df.columns:
        try:
            X = df.drop(['Plan Type', 'Churn'], axis=1, errors='ignore')
            y = df['Plan Type']
        except Exception as e:
            print(f"Error preparing data: {e}. Creating a dummy model...")
            class DummyModel:
                def predict(self, X):
                    return np.array([2] * len(X))

                def decision_function(self, X):
                    return np.array([0] * len(X))

                def predict_proba(self, X):
                    return np.array([[0, 0, 1]] * len(X))

            return DummyModel(), None
    else:
        # If Plan Type is not in the dataset, use a dummy model
        print("Plan Type column not found. Creating a dummy model...")
        class DummyModel:
            def predict(self, X):
                return np.array([2] * len(X))

            def decision_function(self, X):
                return np.array([0] * len(X))

            def predict_proba(self, X):
                return np.array([[0, 0, 1]] * len(X))

        return DummyModel(), None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Scale features
    scaler = StandardScaler()
    try:
        # Try to scale the specified columns
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    except KeyError as e:
        print(f"Warning: Could not scale specified columns: {e}")
        # Fall back to scaling all numeric columns
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Train model - using LinearSVC as in the notebook
    try:
        model = LinearSVC(dual='auto', random_state=101)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training plan recommender LinearSVC: {e}. Creating a dummy model...")
        class DummyModel:
            def predict(self, X):
                return np.array([2] * len(X))

            def decision_function(self, X):
                return np.array([0] * len(X))

            def predict_proba(self, X):
                return np.array([[0, 0, 1]] * len(X))

        return DummyModel(), scaler

    return model, scaler

def train_xgboost_model(df, scaling_cols):
    """Train the XGBoost model as used in the final_insurance_recommender notebook"""
    # Prepare data
    try:
        if 'Churn' in df.columns:
            y = df['Churn']
            X = df.drop("Churn", axis=1)
        else:
            # Assume last column is churn
            y = df.iloc[:, -1]  # Last column is churn
            X = df.iloc[:, :-1]  # All columns except the last one

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
    except Exception as e:
        print(f"Error training XGBoost model: {e}. Creating a dummy model...")
        # Create a dummy model that always predicts 0 (no churn)
        class DummyXGBModel:
            def predict(self, X):
                return np.array([0] * len(X))

            def predict_proba(self, X):
                # Return probabilities [no_churn, churn] with high probability for no churn
                return np.array([[0.9, 0.1]] * len(X))

        return DummyXGBModel(), None

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
