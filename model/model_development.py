import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# 1. Load Wine Dataset
# =====================================================
data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["cultivar"] = data.target   # Classes: 0, 1, 2

print("Wine dataset loaded successfully")
print("Dataset shape:", df.shape)

# =====================================================
# 2. Feature Selection (Choose 6 features)
# =====================================================
selected_features = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "total_phenols",
    "color_intensity"
]

X = df[selected_features]
y = df["cultivar"]

print("\nSelected Features:")
print(selected_features)

# =====================================================
# 3. Handle Missing Values (if any)
# =====================================================
X = X.fillna(X.mean())

# =====================================================
# 4. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 5. Feature Scaling (MANDATORY)
# =====================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 6. Train Model (SVM - Multiclass)
# =====================================================
model = SVC(kernel="rbf", decision_function_shape="ovr")
model.fit(X_train_scaled, y_train)

print("\nModel training completed")

# =====================================================
# 7. Model Evaluation
# =====================================================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nEvaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# 8. Save Model and Scaler
# =====================================================
joblib.dump(model, "wine_cultivar_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully")

# =====================================================
# 9. Reload Model & Test Prediction (No Retraining)
# =====================================================
loaded_model = joblib.load("wine_cultivar_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

sample_input = [[13.2, 2.7, 2.4, 19.0, 2.8, 5.0]]
sample_scaled = loaded_scaler.transform(sample_input)

prediction = loaded_model.predict(sample_scaled)

print("\nSample Prediction Test:")
print(f"Predicted Wine Cultivar: Cultivar {prediction[0] + 1}")