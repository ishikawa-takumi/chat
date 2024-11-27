#!/bin/bash
set -e

echo "Building Docker images..."
docker build -t django-app ./DjangoApp
docker build -t nginx ./Nginx

echo "Deploying Django and Nginx to Kubernetes..."
kubectl apply -f manifests/django-deployment.yaml
kubectl apply -f manifests/nginx-deployment.yaml
kubectl apply -f manifests/nginx-service.yaml

echo "Deployment complete! Access the app at http://<NODE_IP>:30001"
