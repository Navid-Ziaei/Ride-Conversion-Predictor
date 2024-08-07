import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
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

def plot_pca_classification(X_test, y_test, y_pred, save_path):
    # Perform PCA to reduce to two components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    # Create masks for the correctly classified, false positives, and false negatives
    correct_mask = y_test == y_pred
    false_positive_mask = (y_test == 0) & (y_pred == 1)
    false_negative_mask = (y_test == 1) & (y_pred == 0)

    # Plot the PCA components with different colors for each type of classification
    plt.figure(figsize=(10, 7))

    plt.scatter(X_pca[correct_mask, 0], X_pca[correct_mask, 1], c='green', label='Correctly Classified')
    plt.scatter(X_pca[false_positive_mask, 0], X_pca[false_positive_mask, 1], c='red', label='False Positives (Type I Error)')
    plt.scatter(X_pca[false_negative_mask, 0], X_pca[false_negative_mask, 1], c='blue', label='False Negatives (Type II Error)')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Logistic Regression Classification Results')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pca_classification.png'))
    plt.close()

def plot_logistic_regression_weights(model, feature_names, save_path):
    coefficients = model.coef_[0]
    feature_importance = pd.Series(coefficients, index=feature_names)
    feature_importance.sort_values(inplace=True)
    feature_importance.to_csv(save_path + '/feature_importance.csv')

    plt.figure(figsize=(15, 10))
    feature_importance.plot(kind='barh', color=np.where(feature_importance > 0, 'b', 'r'))
    plt.title('Logistic Regression Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Logistic_Regression_feature_importance.png'))
    plt.close()

def plot_xgboost_feature_importance(model, feature_names, save_path):
    df = pd.DataFrame(model.feature_importances_, columns = ['Importance'], index=feature_names)
    df.to_csv(save_path+ 'feature_importance.csv')
    # Sort the DataFrame by importance
    df = df.sort_values(by='Importance', ascending=False)
    top_n = 10
    # Select the top N features
    top_features = df.head(top_n)

    # Plot the feature importances
    plt.figure(figsize=(10, 7))
    plt.bar(top_features.index, top_features['Importance'])
    plt.xlabel('Feature Names')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'top_feature_importance.png'))
    plt.close()

    plt.figure(figsize=(15, 10))
    plot_importance(model)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'XGBoost_feature_importance.png'))
    plt.close()
