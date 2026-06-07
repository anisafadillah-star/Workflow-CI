import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

df = pd.read_csv('MLProject/heart_preprocesing.csv')
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Heart_Disease")

with mlflow.start_run(run_name="logistic_regression_basic"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    print("Training Selesai dengan Autolog lokal!")
