# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Step 1: Load and Prepare the Dataset
# Load dataset (Replace 'path_to_dataset.csv' with your dataset path)
dataset = pd.read_csv('path_to_dataset.csv')

# Step 2: Data Preprocessing
# Data Cleaning: Drop missing values
dataset.dropna(inplace=True)

# Feature Selection (example features)
features = ['packet_size', 'packet_rate', 'session_duration', 'protocol']
X = dataset[features]
y = dataset['label']  # 'label' column indicates whether the traffic is normal or a DDoS attack

# Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Step 3: Model Training and Evaluation
# Evaluation metrics function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# 3.1 XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_metrics = evaluate_model(y_test, y_pred_xgb)
print("XGBoost Metrics:")
print(f"Accuracy: {xgb_metrics[0]:.4f}, Precision: {xgb_metrics[1]:.4f}, Recall: {xgb_metrics[2]:.4f}, F1 Score: {xgb_metrics[3]:.4f}\n")

# 3.2 K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_metrics = evaluate_model(y_test, y_pred_knn)
print("KNN Metrics:")
print(f"Accuracy: {knn_metrics[0]:.4f}, Precision: {knn_metrics[1]:.4f}, Recall: {knn_metrics[2]:.4f}, F1 Score: {knn_metrics[3]:.4f}\n")

# 3.3 Stochastic Gradient Descent (SGD) Classifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
sgd_metrics = evaluate_model(y_test, y_pred_sgd)
print("SGD Metrics:")
print(f"Accuracy: {sgd_metrics[0]:.4f}, Precision: {sgd_metrics[1]:.4f}, Recall: {sgd_metrics[2]:.4f}, F1 Score: {sgd_metrics[3]:.4f}\n")

# 3.4 Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
nb_metrics = evaluate_model(y_test, y_pred_nb)
print("Naive Bayes Metrics:")
print(f"Accuracy: {nb_metrics[0]:.4f}, Precision: {nb_metrics[1]:.4f}, Recall: {nb_metrics[2]:.4f}, F1 Score: {nb_metrics[3]:.4f}\n")

# Step 4: Summary of Results
print("\nSummary of All Model Metrics:")
models = ['XGBoost', 'KNN', 'SGD', 'Naive Bayes']
metrics = [xgb_metrics, knn_metrics, sgd_metrics, nb_metrics]

for model, metric in zip(models, metrics):
    print(f"{model}: Accuracy: {metric[0]:.4f}, Precision: {metric[1]:.4f}, Recall: {metric[2]:.4f}, F1 Score: {metric[3]:.4f}")
