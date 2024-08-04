import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, classification_report

import os


def plot_and_save_results(y_test, y_pred, classifier_name, save_path):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{classifier_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_path, f'{classifier_name}_confusion_matrix.png'))
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(save_path, f'{classifier_name}_classification_report.csv'))

    # Print classification report to console
    print(f'{classifier_name} Classification Report:\n{classification_report(y_test, y_pred)}')


def plot_logistic_regression_weights(model, feature_names, save_path):
    coefficients = model.coef_[0]
    feature_importance = pd.Series(coefficients, index=feature_names)
    feature_importance.sort_values(inplace=True)

    plt.figure(figsize=(10, 7))
    feature_importance.plot(kind='barh', color=np.where(feature_importance > 0, 'b', 'r'))
    plt.title('Logistic Regression Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(save_path, 'Logistic_Regression_feature_importance.png'))
    plt.close()

def plot_xgboost_feature_importance(model, save_path):
    plt.figure(figsize=(10, 7))
    plot_importance(model)
    plt.title('XGBoost Feature Importance')
    plt.savefig(os.path.join(save_path, 'XGBoost_feature_importance.png'))
    plt.close()