#!/bin/bash
set -e

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing Docker..."
sudo apt install -y docker.io

echo "Installing Kubernetes tools..."
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

echo "Initializing Kubernetes cluster..."
sudo kubeadm init --pod-network-cidr=192.168.0.0/16

echo "Setting up kubectl for the current user..."
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

echo "Installing Calico network plugin..."
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

echo "Kubernetes cluster setup complete!"
