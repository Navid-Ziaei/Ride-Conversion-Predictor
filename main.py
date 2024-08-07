import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from src.model.model_training import train_logistic_regression, train_xgboost, train_pytorch_tabular_based
from src.utils import *  # Import utility functions from src.utils
from src.settings import Settings, Paths
from src.data import SnapDataLoader, EDA, DataProcessor
from src.visualization import *  # Import visualization functions from src.visualization


# Initialize settings and load configuration
settings = Settings()
settings.load_settings()

# Initialize paths for saving results and other file operations
paths = Paths(settings=settings)
paths.load_device_paths()

# Load and preprocess data
data_loader = SnapDataLoader(settings, paths)
data_loader.load_data()

# Create a DataProcessor instance to handle preprocessing tasks
data_processor = DataProcessor(data_loader, paths)
# Preprocess data with the specified strategies and extract relevant components
passenger_id, data, target, passenger_id_final_test, data_final_test = (
    data_processor.preprocess(impute_strategy=settings.impute_strategy,
                              missing_threshold=settings.missing_threshold,
                              remove_out_layers=settings.remove_out_layers,
                              scaling_mode=settings.scaling_mode))

# Define sampling strategies to address class imbalance
sampling_methods = {
    'RandomOverSampler': RandomOverSampler(random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'None': None
}

# Set up StratifiedKFold cross-validation with 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

# Iterate over models specified in settings
for model_name in settings.model:
    for sampling_method_name in settings.sampling_method:
        sampler = sampling_methods[sampling_method_name]
        if model_name in ['logistic_regression', 'xgboost']:
            fold_result = []
            result_base_path = paths.result_path + f"{model_name}_{sampling_method_name}/"
            Path(result_base_path).mkdir(parents=True, exist_ok=True)

            # Perform cross-validation
            for idx, (train_index, val_index) in enumerate(skf.split(data.values, target.values)):
                fold_result_path = result_base_path + f"fold{idx + 1}"
                Path(fold_result_path).mkdir(parents=True, exist_ok=True)
                X_train, X_val = data.values[train_index], data.values[val_index]
                y_train, y_val = target.values[train_index], target.values[val_index]

                # Apply the chosen sampling strategy to the training data
                if sampler is None:
                    X_train_resampled, y_train_resampled = X_train, y_train
                else:
                    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

                # Train and evaluate the model based on its type
                if model_name == 'logistic_regression':
                    log_reg, results, y_pred = train_logistic_regression(X_train_resampled, y_train_resampled,
                                                                         X_test=X_val,
                                                                         y_test=y_val,
                                                                         save_path=fold_result_path,
                                                                         sampling_method=sampling_method_name,
                                                                         model_name=model_name,
                                                                         feature_names=data.columns.to_list(),
                                                                         fold=idx)
                elif model_name == 'xgboost':
                    log_reg, results, y_pred = train_xgboost(X_train_resampled, y_train_resampled,
                                                             X_test=X_val,
                                                             y_test=y_val,
                                                             save_path=fold_result_path,
                                                             sampling_method=sampling_method_name,
                                                             model_name=model_name,
                                                             feature_names=data.columns.to_list(),
                                                             fold=idx)

                fold_result.append(results)
            # Aggregate results and save to CSV
            df_result, dict_result = append_fold_results(fold_result, sampling_method_name, model_name=model_name)
            df_result.to_csv(result_base_path + f'result_{sampling_method_name}.csv')
            all_results.extend(dict_result)
        else:
            result_base_path = paths.result_path + f"{model_name}_{sampling_method_name}/"
            Path(result_base_path).mkdir(parents=True, exist_ok=True)
            # Train and evaluate the PyTorch tabular model
            tabular_model, results, y_pred = train_pytorch_tabular_based(data, target,
                                                                         save_path=result_base_path,
                                                                         num_col_names=data_processor.numerical_features,
                                                                         cat_col_names=data_processor.categorical_features,
                                                                         sampling_method=sampling_method_name,
                                                                         model_name=model_name,
                                                                         feature_names=data.columns.to_list(),
                                                                         sampler=sampler)
            # Predict and save the final results
            pred_df = tabular_model.predict(data_final_test)
            final_result = pd.concat([passenger_id_final_test, pred_df], axis=1)
            final_result.to_csv(result_base_path + 'output_file.csv')
            all_results.append(results)

# Save all results to a CSV file
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(paths.result_path + f'all_results.csv')
