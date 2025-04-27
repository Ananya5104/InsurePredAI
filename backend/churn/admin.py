from django.contrib import admin
from django.urls import path
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
from .models import CustomerRecord
import requests
from django.conf import settings

class CustomerRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'age', 'gender', 'earnings', 'claim_amount', 'insurance_plan_amount', 'plan_type', 'churn_probability', 'created_at')
    list_filter = ('gender', 'plan_type', 'credit_score', 'marital_status')
    search_fields = ('id', 'age', 'earnings')
    readonly_fields = ('created_at',)
    list_per_page = 20

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
            'fields': ('churn_probability', 'recommendation')
        }),
        ('System Information', {
            'fields': ('created_at',)
        }),
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('retrain-model/', self.admin_site.admin_view(self.retrain_model_view), name='retrain-model'),
        ]
        return custom_urls + urls

    def retrain_model_view(self, request):
        """View to handle model retraining from admin interface"""
        if request.method == 'POST':
            try:
                # Make a request to the retrain API endpoint
                api_url = request.build_absolute_uri('/api/retrain-model/')

                # Get the session cookie
                session_key = request.COOKIES.get('sessionid')

                response = requests.post(
                    api_url,
                    headers={
                        'Content-Type': 'application/json',
                        'Cookie': f'sessionid={session_key}'
                    }
                )

                if response.status_code == 200:
                    messages.success(request, f"Model retrained successfully: {response.json().get('message', '')}")
                else:
                    messages.error(request, f"Error retraining model: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                messages.error(request, f"Error retraining model: {str(e)}")

            return HttpResponseRedirect(reverse('admin:churn_customerrecord_changelist'))

        context = {
            'title': 'Retrain Model',
            'opts': self.model._meta,
            'app_label': self.model._meta.app_label,
        }
        return TemplateResponse(request, 'admin/retrain_model.html', context)

    def changelist_view(self, request, extra_context=None):
        """Add a button to the changelist view to retrain the model"""
        extra_context = extra_context or {}
        extra_context['show_retrain_button'] = True
        return super().changelist_view(request, extra_context=extra_context)

admin.site.register(CustomerRecord, CustomerRecordAdmin)
