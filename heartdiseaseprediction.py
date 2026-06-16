import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ====================================
# Column Names
# ====================================

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv("heart.csv", names=columns)

# ====================================
# Handle Missing Values
# ====================================

df.replace('?', np.nan, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# ====================================
# Convert Target Column
# ====================================

# 0 = No Disease
# 1,2,3,4 = Disease Present

df['target'] = df['target'].apply(
    lambda x: 1 if x > 0 else 0
)

# ====================================
# Features and Labels
# ====================================

X = df.drop('target', axis=1)
y = df['target']

# ====================================
# Split Dataset
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# Feature Scaling
# ====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================
# Random Forest Model
# ====================================

rf = RandomForestClassifier(random_state=42)

# Train Model
rf.fit(X_train, y_train)

# ====================================
# Test Prediction
# ====================================

pred = rf.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, pred)

# ====================================
# Sample Patient Data
# ====================================

sample = pd.DataFrame(
    [[63,1,1,145,233,1,2,150,0,2.3,3,0,6]],
    columns=X.columns
)

# Scale Sample Data
sample = scaler.transform(sample)

# Predict Disease
prediction = rf.predict(sample)

# ====================================
# Final Output
# ====================================

print("====================================")
print("   Disease Prediction System")
print("====================================")

print("\nDisease: Heart Disease")

if prediction[0] == 1:
    print("Prediction: High chance of heart disease")
else:
    print("Prediction: Low chance of heart disease")

print("\nModel Used: Random Forest")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("====================================")