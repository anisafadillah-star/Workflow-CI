import os
import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    default_filename = "heart_preprocesing.csv"
    file_path = default_filename
    if len(sys.argv) > 1:
        file_path = sys.argv[-1]

    if not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_path = os.path.join(script_dir, default_filename)
        if os.path.exists(alternative_path):
            file_path = alternative_path

    print(f"==================================================")
    print(f"Mencoba memuat dataset dari: {file_path}")
    print(f"==================================================")

    if not os.path.exists(file_path):
        print(f"Error Fatal: File '{file_path}' tidak ditemukan di sistem!")
        print("Pastikan file dataset berada di dalam folder proyek Anda.")
        sys.exit(1)

    df = pd.read_csv(file_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Latihan Credit Scoring")

    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        # Membungkus standarisasi data & algoritma ke dalam satu Pipeline resmi
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)
        print(f"Proses Berhasil! Akurasi Model: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
