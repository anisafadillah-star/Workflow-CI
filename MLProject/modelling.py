import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease")

df = pd.read_csv("heart_preprocesing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():
    full_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    full_pipeline.fit(X_train, y_train)

    accuracy = full_pipeline.score(X_test, y_test)
    print("Accuracy:", accuracy)

    joblib.dump(full_pipeline, "model.pkl")
    print("Model pipeline sukses disimpan ke 'model.pkl'!")
