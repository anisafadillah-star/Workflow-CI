import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if os.path.exists("heart_preprocesing.csv"):
    DATA_PATH = "heart_preprocesing.csv"
elif os.path.exists("MLProject/heart_preprocesing.csv"):
    DATA_PATH = "MLProject/heart_preprocesing.csv"
else:
    DATA_PATH = "Membangun_model/heart_preprocesing.csv"

print(f"Membaca dataset tunggal dari: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=== TRAINING BERHASIL ===")
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
