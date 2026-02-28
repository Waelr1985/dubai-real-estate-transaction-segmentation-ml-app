#!/bin/bash
# Azure Machine Learning Deployment Script
# Ensure you are logged in using `az login` and have set your default workspace.

ENDPOINT_NAME="customer-segmentation-endpoint"
DEPLOYMENT_NAME="blue"

echo "1. Registering the model..."
az ml model create --name segmentation-model --version 1 --path ../models/segmentation_pipeline.pkl

echo "2. Creating the Managed Online Endpoint..."
az ml online-endpoint create --name $ENDPOINT_NAME --auth-mode key

echo "3. Creating the Deployment..."
# Create a YAML configuration for deployment dynamically or point to an existing one.
cat <<EOF > deployment.yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: $DEPLOYMENT_NAME
endpoint_name: $ENDPOINT_NAME
model: azureml:segmentation-model:1
code_configuration:
  code: .
  scoring_script: score.py
environment:
  conda_file: conda.yml
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
instance_type: Standard_DS3_v2
instance_count: 1
EOF

az ml online-deployment create --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME --file deployment.yml --all-traffic

echo "Deployment complete! Endpoint is ready to receive traffic."
