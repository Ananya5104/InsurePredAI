from django.contrib import admin
from django.contrib import messages
from .models import CustomerRecord
from .model_trainer import train_model

def retrain_model_action(modeladmin, request, queryset):
    """Admin action to retrain the model using the data in the database"""
    try:
        # Get all customer records
        all_records = CustomerRecord.objects.all()

        if all_records.count() == 0:
            messages.error(request, "No data available for model retraining.")
            return

        # Prepare data for training
        features = []
        targets = []

        for record in all_records:
            # Skip records without churn value
            if record.churn is None:
                continue

            # Convert data to the format expected by the model trainer
            feature = [
                record.age,
                1 if record.gender == 'M' else 0,  # Convert gender to binary
                record.earnings,
                record.claim_amount,
                record.insurance_plan_amount,
                1 if record.credit_score else 0,  # Convert boolean to binary
                1 if record.marital_status == 'M' else 0,  # Convert marital status to binary
                record.days_passed,
                1 if record.type_of_insurance == 'auto' else 0,  # Auto insurance
                1 if record.type_of_insurance == 'health' else 0,  # Health insurance
                1 if record.type_of_insurance == 'life' else 0,  # Life insurance
                1 if record.plan_type == 'premium' else 0  # Plan type (1 for basic, 2 for premium)
            ]

            features.append(feature)
            targets.append(1 if record.churn == 'Yes' else 0)  # Convert churn to binary

        # Train the model
        train_model(features, targets)

        messages.success(request, f"Model retrained successfully with {len(features)} records.")
    except Exception as e:
        messages.error(request, f"Error retraining model: {str(e)}")

retrain_model_action.short_description = "Retrain model with all available data"

class CustomerRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'age', 'gender', 'earnings', 'claim_amount', 'insurance_plan_amount', 'plan_type', 'churn', 'created_at')
    list_filter = ('gender', 'plan_type', 'credit_score', 'marital_status', 'churn')
    search_fields = ('id', 'age', 'earnings')
    readonly_fields = ('created_at',)
    list_per_page = 20
    actions = [retrain_model_action]

    fieldsets = (
        ('Personal Information', {
            'fields': ('age', 'gender', 'marital_status', 'credit_score')
        }),
        ('Financial Information', {
            'fields': ('earnings', 'claim_amount', 'insurance_plan_amount')
        }),
        ('Insurance Details', {
            'fields': ('type_of_insurance', 'plan_type', 'days_passed')
        }),
        ('Prediction Results', {
            'fields': ('churn', 'churn_probability', 'recommendation')
        }),
        ('System Information', {
            'fields': ('created_at',)
        }),
    )



admin.site.register(CustomerRecord, CustomerRecordAdmin)
