#!/bin/bash
set -e

echo "Deploying application to Kubernetes..."
kubectl apply -f manifests/nginx-deployment.yaml
kubectl apply -f manifests/nginx-service.yaml

echo "Application deployed successfully!"
