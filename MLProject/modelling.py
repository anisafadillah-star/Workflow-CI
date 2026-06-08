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

    file_path = "heart_preprocesing.csv"

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[-1]) and sys.argv[-1].endswith('.csv'):
            file_path = sys.argv[-1]

    if not os.path.exists(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_path = os.path.join(script_dir, "heart_preprocesing.csv")
        if os.path.exists(alternative_path):
            file_path = alternative_path

    if not os.path.exists(file_path):
        sys.exit(1)

    df = pd.read_csv(file_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Latihan Credit Scoring")

    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
