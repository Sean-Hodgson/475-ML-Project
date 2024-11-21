import pandas as pd
import os
import numpy as np
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary libraries are installed
# You can install missing libraries using pip if needed
# For example:
# pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

def load_and_preprocess_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)

    # Remove 'Unnamed' columns if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Encode 'Label' column
    label_mapping = {
        "Benign": 0,
        "FTP-BruteForce": 1,
        "SSH-Bruteforce": 2,
        "DDOS attack-LOIC-UDP": 3,
        "DDOS attack-HOIC": 4,
        "Infilteration": 5,
    }
    df["Label"] = df["Label"].replace(label_mapping)

    # Define feature list
    features_list = [
        "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
        "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean",
        "Pkt Len Std", "Pkt Len Var", "Down/Up Ratio", "Pkt Size Avg", "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
        "Subflow Bwd Pkts", "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    ]

    # Ensure all features are in the DataFrame
    features_list = [feature for feature in features_list if feature in df.columns]

    # Separate features and labels
    X = df[features_list]
    y = df['Label']

    # Handle missing values if any
    if X.isnull().values.any():
        # Fill missing values
        imp = KNNImputer()
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, features_list

def balance_data(X_train, y_train):
    # Apply SMOTE to oversample minority classes
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled

def feature_importance_analysis(model, features_list):
    # Get feature importances
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=features_list)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Plot Feature Importances
    plt.figure(figsize=(12, 8))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

def optimize_hyperparameters(X_train, y_train):
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }

    rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_test, y_scores, class_of_interest):
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]

    # For multiclass, compute PR curve for each class
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])

        plt.plot(recall[i], precision[i], lw=2, label='Class {}'.format(i))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    # File path to the dataset
    filepath = "02_14_21_28_cleaned.csv"

    if os.path.exists(filepath):
        # Load and preprocess data
        X_scaled, y, features_list = load_and_preprocess_data(filepath)

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        # Balance the training data
        X_train_resampled, y_train_resampled = balance_data(X_train, y_train)

        # Verify class distribution after resampling
        print("Resampled training set class distribution:")
        print(Counter(y_train_resampled))

        # Adjust class weights manually
        class_weights = {
            0: 1.0,  # Benign
            1: 2.0,  # FTP-BruteForce
            2: 1.0,  # SSH-Bruteforce
            4: 1.0,  # DDOS attack-HOIC
            5: 5.0   # Infiltration
        }

        # Initialize Random Forest classifier with adjusted class weights
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weights,
            n_jobs=-1
        )

        # Hyperparameter tuning (uncomment to perform)
        # rf_clf = optimize_hyperparameters(X_train_resampled, y_train_resampled)

        # Train the model
        print("Starting model training with SMOTE and adjusted class weights...")
        rf_clf.fit(X_train_resampled, y_train_resampled)

        # Feature importance analysis
        feature_importance_analysis(rf_clf, features_list)

        # Predict on test set
        y_pred = rf_clf.predict(X_test)
        y_proba = rf_clf.predict_proba(X_test)

        # Evaluate the model
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plot_confusion_matrix(cm, classes=rf_clf.classes_, title='Confusion Matrix')

        # Plot Precision-Recall Curve
        plot_precision_recall_curve(y_test, y_proba, class_of_interest=5)

        # Adjust the decision threshold for the Infiltration class
        infiltration_class_index = list(rf_clf.classes_).index(5)
        thresholds = np.linspace(0, 1, 100)

        best_threshold = 0.5
        best_f1 = 0
        for threshold in thresholds:
            y_pred_adjusted = []
            for i in range(len(y_test)):
                if y_proba[i][infiltration_class_index] > threshold:
                    y_pred_adjusted.append(5)
                else:
                    # Predict the class with the highest probability
                    y_pred_adjusted.append(rf_clf.classes_[np.argmax(y_proba[i])])

            report = classification_report(y_test, y_pred_adjusted, output_dict=True)
            f1 = report[str(5)]['f1-score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Best Threshold for Infiltration Class: {best_threshold}")
        print(f"Best F1 Score for Infiltration Class: {best_f1}")

        # Final prediction with best threshold
        y_pred_final = []
        for i in range(len(y_test)):
            if y_proba[i][infiltration_class_index] > best_threshold:
                y_pred_final.append(5)
            else:
                y_pred_final.append(rf_clf.classes_[np.argmax(y_proba[i])])

        # Evaluate the final model
        print("\nFinal Classification Report with Adjusted Threshold:")
        print(classification_report(y_test, y_pred_final, digits=4))

        cm_final = confusion_matrix(y_test, y_pred_final)
        print("Final Confusion Matrix with Adjusted Threshold:")
        print(cm_final)

        # Plot final confusion matrix
        plot_confusion_matrix(cm_final, classes=rf_clf.classes_, title='Final Confusion Matrix')

    else:
        print(f"Data file '{filepath}' not found.")

if __name__ == '__main__':
    main()
