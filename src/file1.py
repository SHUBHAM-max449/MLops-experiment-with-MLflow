import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import dagshub
dagshub.init(repo_owner='SHUBHAM-max449', repo_name='MLops-experiment-with-MLflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/SHUBHAM-max449/MLops-experiment-with-MLflow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

mlflow.set_experiment('YT_MLOps')
mlflow.autolog()


with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)

    accuracy=accuracy_score(y_pred,y_test)
    print(f'accuracy: {accuracy}')

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimator',n_estimators)

    cm=confusion_matrix(y_pred,y_test)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.savefig('confusion-matrix.png')

    mlflow.sklearn.log_model(rf,"RandomForestClassifier")



