import os
import sys
import pandas as pd

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import the model_trainer module
from backend.churn.model_trainer import train_models

# Train the models
print("Starting model training...")
models_dict = train_models()
print("Model training completed!")

# Print the keys in the models_dict
print("Models trained:", list(models_dict.keys()))
