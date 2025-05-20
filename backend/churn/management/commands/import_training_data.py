import os
import csv
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from churn.models import CustomerRecord

class Command(BaseCommand):
    help = 'Import training data from CSV to the database'

    def handle(self, *args, **options):
        # Define the path to the training data CSV
        csv_path = os.path.join(settings.BASE_DIR, 'logs', 'training_data.csv')
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at {csv_path}'))
            return
        
        # Define column names
        column_names = [
            'Age', 'Gender', 'Earnings ($)', 'Claim Amount ($)', 'Insurance Plan Amount ($)',
            'Credit Score', 'Marital Status', 'days_passed', 'Automobile Insurance',
            'Health Insurance', 'Life Insurance', 'Plan Type', 'Churn'
        ]
        
        # Check if the file has headers by reading the first line
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
        
        # If the first line contains only numbers and commas, it's likely there are no headers
        if all(c.isdigit() or c in ',.+-' for c in first_line.replace(',', '')):
            self.stdout.write(self.style.WARNING("Dataset appears to have no headers. Adding column names..."))
            # Read CSV with specified column names
            df = pd.read_csv(csv_path, header=None, names=column_names)
        else:
            # Read CSV normally
            df = pd.read_csv(csv_path)
            
            # If the dataset has numeric column names, rename them
            if not any(col in df.columns for col in column_names):
                self.stdout.write(self.style.WARNING("Dataset has numeric column names. Renaming columns..."))
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
        
        # Count records before import
        initial_count = CustomerRecord.objects.count()
        self.stdout.write(self.style.SUCCESS(f'Found {len(df)} records in CSV file'))
        self.stdout.write(self.style.SUCCESS(f'Current records in database: {initial_count}'))
        
        # Import data
        records_created = 0
        records_skipped = 0
        
        for _, row in df.iterrows():
            try:
                # Map CSV data to CustomerRecord model fields
                gender = 'M' if int(row['Gender']) == 1 else 'F'
                marital_status = 'M' if int(row['Marital Status']) == 1 else 'S'
                
                # Determine insurance type based on the highest value in the insurance columns
                insurance_types = {
                    'auto': int(row['Automobile Insurance']),
                    'health': int(row['Health Insurance']),
                    'life': int(row['Life Insurance'])
                }
                type_of_insurance = max(insurance_types, key=insurance_types.get)
                
                # Determine plan type
                plan_type = 'premium' if int(row['Plan Type']) == 2 else 'basic'
                
                # Determine churn status
                churn_value = 'Yes' if int(row['Churn']) == 1 else 'No'
                
                # Create CustomerRecord object
                CustomerRecord.objects.create(
                    age=int(row['Age']),
                    gender=gender,
                    earnings=float(row['Earnings ($)']),
                    claim_amount=float(row['Claim Amount ($)']),
                    insurance_plan_amount=float(row['Insurance Plan Amount ($)']),
                    credit_score=bool(int(row['Credit Score'])),
                    marital_status=marital_status,
                    days_passed=int(row['days_passed']),
                    type_of_insurance=type_of_insurance,
                    plan_type=plan_type,
                    churn=churn_value,
                    churn_probability=1.0 if churn_value == 'Yes' else 0.0,
                    recommendation='Imported from training data'
                )
                records_created += 1
                
                # Print progress every 100 records
                if records_created % 100 == 0:
                    self.stdout.write(self.style.SUCCESS(f'Imported {records_created} records...'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error importing record: {e}'))
                records_skipped += 1
        
        # Final count
        final_count = CustomerRecord.objects.count()
        self.stdout.write(self.style.SUCCESS(f'Import complete!'))
        self.stdout.write(self.style.SUCCESS(f'Records created: {records_created}'))
        self.stdout.write(self.style.SUCCESS(f'Records skipped: {records_skipped}'))
        self.stdout.write(self.style.SUCCESS(f'Total records in database: {final_count}'))
