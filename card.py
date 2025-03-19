import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------- STEP 1: Load Data from the Web ----------------------------

def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ---------------------------- STEP 2: Data Preprocessing ----------------------------
print("Dataset Overview:")
print(df.head())

# Standardize the 'Amount' column
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

# Drop 'Time' column as it is not useful for modeling
df.drop(columns=['Time'], inplace=True)

# Separate features and labels
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Labels (0 = Normal, 1 = Fraud)

# ---------------------------- STEP 3: Train Unsupervised Anomaly Detection Models ----------------------------

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X)
y_pred_iso = iso_forest.predict(X)
y_pred_iso = [1 if p == -1 else 0 for p in y_pred_iso]  # Convert to fraud (1) and normal (0)

# Train One-Class SVM
oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
oc_svm.fit(X)
y_pred_svm = oc_svm.predict(X)
y_pred_svm = [1 if p == -1 else 0 for p in y_pred_svm]  # Convert to fraud (1) and normal (0)

# ---------------------------- STEP 4: Evaluate Models ----------------------------
print("\nIsolation Forest Performance:")
print(classification_report(y, y_pred_iso))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_iso))

print("\nOne-Class SVM Performance:")
print(classification_report(y, y_pred_svm))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_svm))

# ---------------------------- STEP 5: Visualization ----------------------------
sns.countplot(x=y)
plt.title("Class Distribution (Normal vs Fraud)")
plt.show()

sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
