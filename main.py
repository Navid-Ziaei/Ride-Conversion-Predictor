
from src.utils import *
from src.settings import Settings, Paths
from src.data import SnapDataLoader, EDA, DataProcessor
from src.visualization import *
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()


# eda_analyzer = EDA(paths=paths)
# eda_analyzer.run_eda()


data_loader = SnapDataLoader(settings, paths)
data_loader.load_data()

# Preprocess data
data_processor = DataProcessor()
df_train = data_loader.get_train_data()
target_column = 'ride (target)'  # Replace with your target column name

X_train, X_val, y_train, y_val = data_processor.preprocess(df_train, target_column)

# Train and evaluate Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_val)
accuracy_log_reg = accuracy_score(y_val, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {accuracy_log_reg}')
plot_and_save_results(y_val, y_pred_log_reg, 'Logistic_Regression', paths.result_path)

# Plot Logistic Regression feature importance
plot_logistic_regression_weights(log_reg, data_processor.processed_col_names, paths.result_path)

# Train and evaluate XGBoost Classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_val)
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
print(f'XGBoost Classifier Accuracy: {accuracy_xgb}')
plot_and_save_results(y_val, y_pred_xgb, 'XGBoost', paths.result_path)
plot_xgboost_feature_importance(log_reg, paths.result_path)
