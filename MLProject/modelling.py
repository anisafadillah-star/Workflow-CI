import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# PENDETEKSI JALUR DATASET (Agar aman saat jalan di GitHub Actions)
if os.path.exists("heart_preprocesing.csv"):
    DATA_PATH = "heart_preprocesing.csv"
elif os.path.exists("MLProject/heart_preprocesing.csv"):
    DATA_PATH = "MLProject/heart_preprocesing.csv"
else:
    DATA_PATH = "heart_preprocesing.csv"

print(f"Membaca dataset dari jalur: {DATA_PATH}")

# 1. Membaca Dataset Berupa File Tunggal
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target'])
y = df['target']

# 2. Membagi Data Training & Testing (Split 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standarisasi Fitur Menggunakan StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Menjalankan Training dengan Logging MLflow Secara Manual
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    
    # Menghitung akurasi akhir untuk log konsol GitHub
    accuracy = model.score(X_test, y_test.values.ravel())
    print("=== TRAINING BERHASIL ===")
    print(f"Training Selesai Berhasil! Akurasi Model: {accuracy:.4f}")
