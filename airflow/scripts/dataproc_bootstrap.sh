#!/bin/bash
set -e

# Install Python 3 pip  
apt-get install -y python3-pip

# Copy the requirements file from GCS to the cluster node
gsutil cp gs://churn-prediction-ai-agent/code/cloud_requirements.txt .

# Install the cloud requirements
pip install -r cloud_requirements.txt
