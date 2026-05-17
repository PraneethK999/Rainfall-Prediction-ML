# ==========================================
# Rainfall Prediction using Machine Learning
# Australian Weather Dataset



# 1. Import Libraries
    # ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv("data/weatherAUS.csv")

# ==========================================
# 3. Display Dataset Information
# ==========================================

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# 4. Data Cleaning
# ==========================================

# Drop rows where target variable is missing
df = df.dropna(subset=['RainTomorrow'])

# ==========================================
# 5. Feature and Target Selection
# ==========================================

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# ==========================================
# 6. Separate Numerical and Categorical Columns
# ==========================================

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object']).columns

# ==========================================
# 7. Preprocessing Pipelines
# ==========================================

# Numerical Pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ==========================================
# 8. Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 9. Logistic Regression Model
# ==========================================

logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train Model
logistic_pipeline.fit(X_train, y_train)

# Predictions
y_pred_lr = logistic_pipeline.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

# ==========================================
# 10. Random Forest Model
# ==========================================

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train Model
rf_pipeline.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# 12. Classification Report
# ==========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# 13. Compare Model Accuracy
# ==========================================

models = ['Logistic Regression', 'Random Forest']

accuracies = [lr_accuracy, rf_accuracy]

plt.figure(figsize=(6,4))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# ==========================================
# 14. Feature Importance (Random Forest)
# ==========================================

rf_model = rf_pipeline.named_steps['classifier']

print("\nRandom Forest Model:")
print(rf_model)

# ==========================================
# 15. Sample Prediction
# ==========================================

sample_prediction = rf_pipeline.predict(X_test.iloc[:5])

print("\nSample Predictions:")
print(sample_prediction)


# 16. Final Output

print("\nProject Completed Successfully")