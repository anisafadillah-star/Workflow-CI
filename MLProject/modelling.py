import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Tracking lokal
mlflow.set_tracking_uri("file:./mlruns")

# Nama experiment
mlflow.set_experiment("Heart_Disease")

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv("heart_preprocesing.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training model
with mlflow.start_run():

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print("Accuracy :", accuracy)
