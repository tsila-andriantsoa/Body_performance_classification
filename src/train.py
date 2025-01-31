#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import joblib

import os

RANDOM_STATE = 42

# Get absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data path
DATA_PATH = os.path.join(PROJECT_DIR, "data", "bodyPerformance.csv")

# Define model path
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "pipeline_baseline.pkl")


# ### Data importation
data = pd.read_csv(DATA_PATH)
print('Data importation step done!')

# ### Data preparation
# renames columns
new_col_names = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_%', 'diastolic',
       'systolic', 'gripForce', 'sit_and_bend_forward_cm', 'sit_ups_counts',
       'broad_jump_cm', 'class']
data.columns = new_col_names

# split data into train validation and test
df_full_train, df_test = train_test_split(data, train_size=.8, random_state = RANDOM_STATE)
df_full_train.shape, df_test.shape

# reset data index
df_full_train.reset_index(drop = True, inplace = True)
df_test.reset_index(drop = True, inplace = True)
print('Data preparation step done!')

# ### Data preprocessing
# check NA values
df_full_train.isnull().sum()
# drop duplicated rows
df_full_train.drop_duplicates(inplace = True)
# check data type
df_full_train.info()
print('Data preprocessing step done!')

# ### EDA

# ##### Numerical features
numerical_features = df_full_train.select_dtypes(include = ['float']).columns.tolist()

# ###### Statistics Summary
df_full_train[numerical_features].describe().T

# ###### Univariate analysis

# ###### Outliers analysis
for col in numerical_features:
    # Compute Q99
    Q99 = df_full_train[col].quantile(0.99)    
    # Compute the mode
    mode_value = df_full_train[col].mode()[0]
    # Replace outliers with the mode
    df_full_train[col] = df_full_train[col].apply(lambda x: mode_value if x > Q99 else x)

# ###### Bivariate analysis

# ###### Multivariate analysis

# ##### Target feature analysis
print('EDA step done!')

# ### Model training

# preparing target encoding
labelencoder = LabelEncoder()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore'))
])

# Combine preprocessors in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, make_column_selector(dtype_include=['float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])


df_full_train['class'] = labelencoder.fit_transform(df_full_train['class'])
X_full_train = df_full_train.drop(columns = ['class'])
y_full_train = df_full_train['class']

X_train, X_validation, y_train, y_validation = train_test_split(X_full_train, y_full_train, train_size=.8)

# Get best parameters for best model from model tunning
xgb_eta = 0.1
xgb_n_estimators = 50
xgb_max_depth = 10

xgb_model = xgb.XGBClassifier(
                eta = xgb_eta, 
                max_depth = xgb_max_depth,
                min_child_weight = 1,
                objective = 'multi:softmax',
                nthread = 12,
                random_state = RANDOM_STATE,
                verbosity = 1,
                n_estimators = xgb_n_estimators
            )
# define pipeline : preprocessor for features transformation and classifier for model training
pipeline_baseline = Pipeline([
    ('preprocessor',  preprocessor),
    ('classifier', xgb_model)     
])

pipeline_baseline.fit(X_full_train, y_full_train)
print('Model training step done!')

# ### Evaluate baseline model

df_test['class'] = labelencoder.fit_transform(df_test['class'])
X_test = df_test.drop(columns = ['class'])
y_test = df_test['class']

y_test_pred = pipeline_baseline.predict(X_test)
print(f'F1 score on test data : {f1_score(y_test, y_test_pred, average = "weighted")}')
print('Model evaluation step done!')

# ### Dump baseline model
# Save final model (best model) and other utilities
with open(MODEL_PATH, 'wb') as f:
    joblib.dump(pipeline_baseline, f)
    print('Best model saved!')
    
