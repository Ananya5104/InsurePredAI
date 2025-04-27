from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from django.shortcuts import render
from.models import CustomerRecord
import json
from .utils import get_comprehensive_analysis
from .model_utils import retrain_model_from_csv
from .model_trainer import train_models
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import CustomerRecord
from .serializers import CustomerRecordSerializer
import os
import csv
from rest_framework import status
from django.conf import settings

# @api_view(['POST'])
# def predict_and_save(request):
#     features = request.data.get("features")
#     raw_data = request.data.get("raw_data")

#     if not features or not raw_data:
#         return Response({"error": "Invalid data"}, status=400)

#     # === ML Prediction (based on features) ===
#     churn_prob = your_model.predict(features)
#     recommendation = your_model.get_recommendation(features)

#     # === Save raw_data + prediction ===
#     raw_data["churn_probability"] = churn_prob
#     raw_data["recommendation"] = recommendation

#     serializer = CustomerRecordSerializer(data=raw_data)
#     if serializer.is_valid():
#         serializer.save()
#         return Response({
#             "churn_probability": churn_prob,
#             "recommendation": recommendation
#         })
#     else:
#         return Response(serializer.errors, status=400)

@api_view(['POST'])
def predict(request):
    try:
        data = request.data

        # Check for required keys
        features = data.get("features")
        raw_data = data.get("raw_data")

        if not features or not isinstance(features, list):
            return Response({"error": "'features' should be a list"}, status=400)

        if not raw_data or not isinstance(raw_data, dict):
            return Response({"error": "Missing or invalid 'raw_data'"}, status=400)

        # 🔍 Perform model prediction
        try:
            result = get_comprehensive_analysis(features)
            churn_data = result.get("churn_analysis", {})
            churn_prob = churn_data.get("churn_probability", 0.0)
            recommendation = churn_data.get("recommendation", "No recommendation.")
        except Exception as model_error:
            print(f"Error in model prediction: {str(model_error)}")
            # Provide a default result if the model fails
            result = {
                "churn_analysis": {
                    "churn_probability": 0.3,
                    "is_churn_risk": False,
                    "recommendation": "Unable to determine churn risk. Please check the input data."
                },
                "plan_recommendation": {
                    "recommended_plan": 2,
                    "recommended_plan_name": "Standard",
                    "current_plan": "Standard",
                    "plan_message": "Recommend the Standard plan based on the customer profile."
                },
                "customer_recommendations": {
                    "General": ["Unable to generate personalized recommendations."]
                }
            }
            churn_prob = 0.3
            recommendation = "Unable to determine churn risk. Please check the input data."

        # ➕ Add prediction data to raw_data
        raw_data["churn_probability"] = float(churn_prob)
        raw_data["recommendation"] = recommendation

        # Append to training data CSV
        try:
            csv_path = os.path.join(settings.BASE_DIR, 'logs', 'training_data.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Create folder if not exists

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # Convert churn_prob to binary (0 or 1) for training data
                churn_binary = 1 if float(churn_prob) > 0.5 else 0
                writer.writerow(features + [churn_binary])
        except Exception as csv_error:
            print(f"Error appending to CSV: {str(csv_error)}")
            # Continue even if CSV append fails

        # ✅ Save raw_data to the database
        try:
            # Ensure data types match the model
            if "credit_score" in raw_data:
                raw_data["credit_score"] = bool(raw_data["credit_score"])

            serializer = CustomerRecordSerializer(data=raw_data)
            if serializer.is_valid():
                serializer.save()
            else:
                print(f"Serializer errors: {serializer.errors}")
                # Continue even if database save fails
        except Exception as db_error:
            print(f"Error saving to database: {str(db_error)}")
            # Continue even if database save fails

        # ✅ Return only prediction result
        return Response(result)

    except json.JSONDecodeError:
        return Response({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        print(f"Unexpected error in predict view: {str(e)}")
        return Response({"error": str(e)}, status=400)

def prediction_form(request):
    return render(request, "prediction_form.html")


def save_customer_data(data, churn_prob, recommendation):
    """
    This function is no longer used. Customer data is saved in the predict view.
    Kept for reference only.
    """
    try:
        from django.db.models import Max

        # Get the current max id
        max_id = CustomerRecord.objects.aggregate(Max('id'))['id__max'] or 0
        next_id = max_id + 1

        # Create the customer record
        CustomerRecord.objects.create(
            age=data['age'],
            gender=data['gender'],
            earnings=data['earnings'],
            claim_amount=data['claim_amount'],
            insurance_plan_amount=data['insurance_plan_amount'],
            credit_score=data['credit_score'],
            marital_status=data['marital_status'],
            days_passed=data['days_passed'],
            type_of_insurance=data['type_of_insurance'],
            plan_type=data['plan_type'],
            churn_probability=float(churn_prob),
            recommendation=recommendation
        )
        return True
    except Exception as e:
        print(f"Error saving customer data: {str(e)}")
        return False

@api_view(['POST'])
def retrain_model_api(request):
    try:
        # Paths relative to the project root (where manage.py is)
        dataset_path = os.path.join(settings.BASE_DIR, 'logs', 'training_data.csv')
        model_output_path = os.path.join(settings.BASE_DIR, 'models', 'churn_model.pkl')

        if not os.path.exists(dataset_path):
            return Response({'error': 'training_data.csv not found.'}, status=400)

        try:
            # Retrain and save model
            message = retrain_model_from_csv(dataset_path, model_output_path)
            return Response({'message': message}, status=200)
        except Exception as train_error:
            print(f"Error retraining model: {str(train_error)}")
            return Response({'error': str(train_error)}, status=500)

    except Exception as e:
        print(f"Unexpected error in retrain_model_api: {str(e)}")
        return Response({'error': str(e)}, status=500)