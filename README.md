# 🚀 End-to-End MLOps Pipeline: ML Model Deployment on Kubernetes (AKS)

## 📌 Overview

This project demonstrates a complete **Machine Learning Operations (MLOps) pipeline**, covering the full lifecycle of an ML model from **data preprocessing → model training → deployment → monitoring → retraining**.

The system is designed using **DevOps + MLOps best practices** to ensure scalability, reliability, and automation.

---

## 🧠 Problem Statement

Predict customer churn based on input features such as age, balance, and tenure.  
The model is deployed as a **REST API** for real-time inference.

---

## 🏗️ Architecture
User Request
↓
FastAPI (ML API)
↓
Docker Container
↓
Kubernetes (AKS)
↓
Monitoring (Prometheus + Grafana)
↓
Drift Detection → Retraining Pipeline



---

## ⚙️ Tech Stack

- **Programming**: Python
- **ML Model**: XGBoost / Scikit-learn
- **API**: FastAPI
- **Containerization**: Docker
- **Orchestration**: Kubernetes (AKS)
- **CI/CD**: GitHub Actions / Jenkins
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Azure (AKS, ACR)

---

## 📂 Project Structure
mlops-project/
│
├── app.py # FastAPI app
├── train_model.py # Model training script
├── model.joblib # Trained model
├── scaler.joblib # Preprocessing pipeline
├── requirements.txt # Dependencies
├── Dockerfile # Container config
├── deployment.yaml # Kubernetes deployment
├── service.yaml # Kubernetes service
├── .github/workflows/ # CI/CD pipeline
└── README.md


---

## 🔄 ML Lifecycle

### 1️⃣ Data Preprocessing
- Data cleaning and feature engineering
- Scaling using StandardScaler

### 2️⃣ Model Training
- Model trained using XGBoost / Linear Regression
- Evaluated using accuracy metrics
- Saved using `joblib`

python train_model.py

Model Serving (API):
FastAPI used to expose model as REST API
Endpoint: /predict
uvicorn app:app --reload  ---cmd to use the service

Containerization
Docker used to package application + model
docker build -t mlops-app .
docker run -p 8000:8000 mlops-app

CI/CD Pipeline
Automated pipeline using GitHub Actions:
Code checkout
Run tests
Build Docker image
Push to registry
Deploy to Kubernetes
6️⃣ Deployment on Kubernetes (AKS)
Deployment using YAML manifests
Auto-scaling using HPA
Load balancing via Kubernetes Service
cmd:
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

Monitoring & Logging
Prometheus collects metrics
Grafana visualizes:
CPU usage
API latency
Request count
8️⃣ Model Monitoring (MLOps)
Track:
Prediction accuracy
Input data distribution
Detect:
Data drift
Performance degradation

Retraining Pipeline

When:

Data drift detected
Accuracy drops below threshold
New Data → Retrain Model → Save Model → Deploy New Version

Author:

Sourabh Dekate
DevOps & MLOps Engineer

