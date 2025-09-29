#!/bin/bash

# OCAD System Startup Script

set -e

echo "Starting ORAN CFM-Lite AI Anomaly Detection System..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create config directory if it doesn't exist
mkdir -p config

# Copy example config if local config doesn't exist
if [ ! -f "config/local.yaml" ]; then
    echo "Creating local configuration from example..."
    cp config/example.yaml config/local.yaml
    echo "Please edit config/local.yaml to match your environment"
fi

# Start the system
echo "Starting OCAD system..."
python -m ocad.api.main

echo "OCAD system started successfully!"
echo "API available at http://localhost:8080"
