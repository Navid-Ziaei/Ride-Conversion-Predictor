#!/usr/bin/env python
# coding: utf-8

# In[9]:
import pandas as pd

from src.utils import *
from src.settings import Settings, Paths
from src.data import SnapDataLoader, EDA




# # Initialization
# The code snippet initializes a `Settings` object and loads various settings into it. Then, it creates a `Paths` object, passing the previously created `settings` object to it. Finally, it calls a method to load device-specific paths based on the settings loaded earlier.
# 

# In[10]:


settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()


# # Loading and Previewing the Data

# In[11]:


eda_analyzer = EDA(paths=paths)
print(f"Number of train samples : {eda_analyzer.train_df.shape[0]}")
print(f"Number of test samples : {eda_analyzer.train_df.shape[0]}")

eda_analyzer.train_df.info()  
eda_analyzer.test_df.info() 


# # Missing Values

# In[12]:


eda_analyzer.check_missing_values()


# # Summary Statistics

# In[13]:


combined_summary = eda_analyzer.summary_statistics()
combined_summary


# # Distribution of Numerical Features

# In[14]:


numerical_features = ['approximate_distance_meter', 'final_price', 'second_destination_final_price',
                      'round_ride_final_price', 'days_since_passenger_first_ride',
                      'days_since_passenger_first_request']
eda_analyzer.analyze_non_zero_distributions()
eda_analyzer.plot_numerical_distributions(numerical_features, mode='train')
eda_analyzer.plot_numerical_boxplots(numerical_features, mode='train')


# In[15]:


eda_analyzer.plot_numerical_distributions(numerical_features, mode='test')
eda_analyzer.plot_numerical_boxplots(numerical_features, mode='test')


# # Distribution of Categorical Features

# In[16]:


categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                                'requested_service_type', 'in_hurry_enabled', 'treatment_group', 'ride (target)']
eda_analyzer.plot_categorical_distributions(categorical_features)


# # Correlation Analysis
# ### 1- fix the date time
# chane the time to day of week and time of day

# In[17]:


eda_analyzer.preprocess_datetime()
eda_analyzer.train_df


# ### 2- Categorical data label encoding

# In[18]:


categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                                'requested_service_type', 'in_hurry_enabled', 'treatment_group']
eda_analyzer.encode_categorical_features(categorical_features)
eda_analyzer.train_df


# ### 3- Calculate correlation

# In[19]:


numerical_features = ['approximate_distance_meter', 'final_price', 'second_destination_final_price',
                      'round_ride_final_price', 'days_since_passenger_first_ride',
                      'days_since_passenger_first_request', 'new_origin_latitude', 'new_origin_longitude'
]
categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                        'requested_service_type', 'in_hurry_enabled', 'treatment_group', 'ride (target)']
target_features = ['ride (target)']
date_time_features = ['request_datetime']
eda_analyzer.correlation_analysis(numerical_features, categorical_features)


# ### 4- pointbiserial correlation

# In[20]:


eda_analyzer.pointbiserial_correlation(numerical_features)


# ### 5- CHi Square test

# In[21]:


eda_analyzer.chi_square_test(categorical_features)


# ### 6- Regression based feature importance

# In[22]:


eda_analyzer.correlation_with_target()


# # PCA

# In[36]:


numerical_features = ['approximate_distance_meter', 'final_price', 'second_destination_final_price',
                      'round_ride_final_price', 'days_since_passenger_first_ride',
                      'days_since_passenger_first_request', 'new_origin_latitude', 'new_origin_longitude'
]
categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                        'requested_service_type', 'in_hurry_enabled', 'treatment_group']

selected_features = categorical_features + numerical_features
eda_analyzer.plot_pca(eda_analyzer.train_df.copy(), selected_features, file_name='pca_before_outlier_removal')


# # Out-layer detection

# In[35]:


numerical_features = ['approximate_distance_meter', 'final_price', 'second_destination_final_price',
                      'round_ride_final_price', 'days_since_passenger_first_ride',
                      'days_since_passenger_first_request', 'new_origin_latitude', 'new_origin_longitude'
]
categorical_features = ['waiting_time_enabled', 'for_friend_enabled', 'is_voucher_used', 'intercity',
                        'requested_service_type', 'in_hurry_enabled', 'treatment_group']

selected_features = categorical_features + numerical_features
eda_analyzer.detect_outliers_with_dbscan(selected_features, eps=1, min_samples=5)


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
data = eda_analyzer.train_df[eda_analyzer.train_df['outlier']==False]
data.drop('outlier', axis=1, inplace=True)

eda_analyzer.plot_pca(data.copy(), selected_features, file_name='pca_after_outlier_removal')
        
        



# In[ ]:




