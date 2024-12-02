import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Function to preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Encode 'Label' column
    label_mapping = {
        "Benign": 0,
        "FTP-BruteForce": 1,
        "SSH-Bruteforce": 2,
        "DDOS attack-LOIC-UDP": 3,
        "DDOS attack-HOIC": 4,
        "Infilteration": 5
    }
    df["Label"] = df["Label"].replace(label_mapping)

    # Define features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to evaluate a classifier
def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name):
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classification Report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load dataset
    filepath = "02_14_21_28_cleaned.csv"
    if not os.path.exists(filepath):
        print(f"File '{filepath}' not found.")
        return

    # Preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Define classifiers to compare
    classifiers = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    }

    # Evaluate each classifier
    for name, clf in classifiers.items():
        evaluate_classifier(clf, X_train, X_test, y_train, y_test, name)

if __name__ == "__main__":
    main()