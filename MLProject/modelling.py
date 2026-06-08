import os
import sys
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = "heart_preprocessing.csv"

    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        file_path = sys.argv[3]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_path = os.path.join(script_dir, "heart_preprocessing.csv")
        if os.path.exists(alternative_path):
            file_path = alternative_path

    if not os.path.exists(file_path):
        sys.exit(1)

    df = pd.read_csv(file_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    input_example = X_train[0:5]
    
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 37

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Latihan Credit Scoring")
    
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
