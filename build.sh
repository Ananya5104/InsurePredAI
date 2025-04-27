#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements-render.txt

# Create necessary directories
mkdir -p backend/logs
mkdir -p backend/models

# Copy the training data to the logs directory if it doesn't exist
if [ ! -f backend/logs/training_data.csv ]; then
    cp training_data.csv backend/logs/training_data.csv
fi

# Collect static files
cd backend
python manage.py collectstatic --no-input
python manage.py migrate
