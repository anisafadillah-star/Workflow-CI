import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    mlflow.autolog()

    df = pd.read_csv("namadataset_preprocessing/heart_preprocesing.csv")
    
    # Memisahkan Fitur dan Target (pastikan kolom 'target' ada di file heart.csv kamu)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train.values.ravel())

        accuracy = model.score(X_test, y_test)
        print(f"Retraining Berhasil! Akurasi Model Baru: {accuracy:.4f}")
