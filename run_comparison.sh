#!/bin/bash

# Data Comparison Test Runner
# This script sets up the environment and runs the data comparison test

echo "=========================================="
echo "Data Comparison Test Setup"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Make the Python script executable
chmod +x data_comparison_test.py

echo "=========================================="
echo "Starting Data Comparison Test"
echo "=========================================="

# Run the comparison test
python3 data_comparison_test.py

echo "=========================================="
echo "Test Execution Complete"
echo "=========================================="

# Deactivate virtual environment
deactivate

echo "Check the generated report files for results."