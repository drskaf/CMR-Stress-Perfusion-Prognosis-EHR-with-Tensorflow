import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
from utils import patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
from keras.models import model_from_json, load_model
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# Load test data
test_data = pd.read_csv('test_data.csv')

# Define columns
categorical_col_list = ['Positive_perf','Positive_LGE','Chronic_kidney_disease','Hypertension', 'Gender', 'Heart_failure' ]
numerical_col_list= ['Age']
PREDICTOR_FIELD = 'Event'

processed_df = preprocess_df(test_data, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 1
survival_test_ds = df_to_dataset(processed_df, PREDICTOR_FIELD, batch_size=batch_size)

# Create categorical features
vocab_file_list = build_vocab_files(test_data, categorical_col_list)
from student_utils import create_tf_categorical_feature_cols
tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list)

# create numerical features
def create_tf_numerical_feature_cols(numerical_col_list, test_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        tf_numeric_feature = tf.feature_column.numeric_column(c)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

tf_cont_col_list = create_tf_numerical_feature_cols(numerical_col_list, test_data)

# Create feature layer
claim_feature_columns = tf_cat_col_list + tf_cont_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

with open('fcn_b.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
survival_model = pickle.load(open('fcn_b.pkl', 'rb'))

# Predict with model
preds = survival_model.predict(survival_test_ds)
pred_test = []
for p in preds:
    pred = np.argmax(p)
    pred_test.append(pred)
print(pred_test[:5])
survival_yhat = list(test_data['Event'].values)
print(survival_yhat[:5])

prob_outputs = {
    "pred": pred_test,
    "actual_value": survival_yhat
}
prob_output_df = pd.DataFrame(prob_outputs)
print(prob_output_df.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test))
print('ROCAUC score:',roc_auc_score(survival_yhat, pred_test))
print('Accuracy score:',accuracy_score(survival_yhat, pred_test))
print('F1 score:',f1_score(survival_yhat, pred_test))
