from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae

# Load dataset
survival_df = pd.read_csv('survival_final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_df['Diabetes_mellitus'] = survival_df['Diabetes_mellitus_(disorder)']
survival_df['Cerebrovascular_accident'] = survival_df['Cerebrovascular_accident_(disorder)']
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)']
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)']
print(survival_df.head())

# Define columns
categorical_col_list = [ 'patient_GenderCode', 'Essential_hypertension',  'Dyslipidaemia',  'Positive_LGE', 'Positive_perf', 'Diabetes_mellitus', 'Cerebrovascular_accident', 'Heart_failure', 'Chronic_kidney_disease']
numerical_col_list = [ 'Age_on_20.08.2021' ]
PREDICTOR_FIELD = 'duration'

# Split data
from student_utils import patient_dataset_splitter
d_train, d_val, d_test = patient_dataset_splitter(survival_df, 'patient_TrustNumber')
assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)
print("Test passed for number of total rows equal!")

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 128
survival_train_ds = df_to_dataset(d_train, PREDICTOR_FIELD, batch_size=batch_size)
survival_val_ds = df_to_dataset(d_val, PREDICTOR_FIELD, batch_size=batch_size)
survival_test_ds = df_to_dataset(d_test, PREDICTOR_FIELD, batch_size=batch_size)

# We use this sample of the dataset to show transformations later
survival_batch = next(iter(survival_train_ds))[0]
def demo(feature_column, example_batch):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))

# Create categorical features
vocab_file_list = build_vocab_files(d_train, categorical_col_list)
from student_utils import create_tf_categorical_feature_cols
tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list)

# Test a batch
test_cat_var1 = tf_cat_col_list[0]
print("Example categorical field:\n{}".format(test_cat_var1))
demo(test_cat_var1, survival_batch)

# Create numerical features
from student_utils import create_tf_numeric_feature
def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list
tf_cont_col_list = create_tf_numerical_feature_cols(numerical_col_list, d_train)

# Test a batch
test_cont_var1 = tf_cont_col_list[0]
print("Example continuous field:\n{}\n".format(test_cont_var1))
demo(test_cont_var1, survival_batch)
