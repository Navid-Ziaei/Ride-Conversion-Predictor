import pandas as pd


def append_fold_results(fold_result, sampling_method_name, model_name):
    df_result = pd.DataFrame(fold_result)
    mean_results = {
        'model': model_name,
        'sampling_method': sampling_method_name,
        'fold': 'avg',
        'accuracy': df_result['accuracy'].mean(),
        'precision': df_result['precision'].mean(),
        'recall': df_result['recall'].mean(),
        'f1': df_result['f1'].mean()
    }

    std_results = {
        'model': model_name,
        'sampling_method': sampling_method_name,
        'fold': 'std',
        'accuracy': df_result['accuracy'].std(),
        'precision': df_result['precision'].std(),
        'recall': df_result['recall'].std(),
        'f1': df_result['f1'].std()
    }

    fold_result.append(mean_results)
    fold_result.append(std_results)
    df_result = pd.DataFrame(fold_result)

    return df_result, [mean_results, std_results]