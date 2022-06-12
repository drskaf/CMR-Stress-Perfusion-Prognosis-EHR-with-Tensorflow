import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
from utils import create_tf_categorical_feature_cols, patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
from keras.models import model_from_json, load_model
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# Load test data
test_data = pd.read_csv('test_data.csv')

test_data['p_basal_anterior'] = test_data['p_basal_anterior'].astype(str)
test_data['p_basal_anteroseptum'] = test_data['p_basal_anteroseptum'].astype(str)
test_data['p_mid_anterior'] = test_data['p_mid_anterior'].astype(str)
test_data['p_mid_anteroseptum'] = test_data['p_mid_anteroseptum'].astype(str)
test_data['p_apical_anterior'] = test_data['p_apical_anterior'].astype(str)
test_data['p_apical_septum'] = test_data['p_apical_septum'].astype(str)
test_data['p_basal_inferolateral'] = test_data['p_basal_inferolateral'].astype(str)
test_data['p_basal_anterolateral'] = test_data['p_basal_anterolateral'].astype(str)
test_data['p_mid_inferolateral'] = test_data['p_mid_inferolateral'].astype(str)
test_data['p_mid_anterolateral'] = test_data['p_mid_anterolateral'].astype(str)
test_data['p_apical_lateral'] = test_data['p_apical_lateral'].astype(str)
test_data['p_basal_inferoseptum'] = test_data['p_basal_inferoseptum'].astype(str)
test_data['p_basal_inferior'] = test_data['p_basal_inferior'].astype(str)
test_data['p_mid_inferoseptum'] = test_data['p_mid_inferoseptum'].astype(str)
test_data['p_mid_inferior'] = test_data['p_mid_inferior'].astype(str)
test_data['p_apical_inferior'] = test_data['p_apical_inferior'].astype(str)

# Define columns
categorical_col_list = ['p_basal_anterior', 'p_basal_anteroseptum', 'p_mid_anterior', 'p_mid_anteroseptum', 'p_apical_anterior',
     'p_apical_septum','p_basal_inferolateral', 'p_basal_anterolateral', 'p_mid_inferolateral', 'p_mid_anterolateral',
     'p_apical_lateral','p_basal_inferoseptum', 'p_basal_inferior', 'p_mid_inferoseptum', 'p_mid_inferior', 'p_apical_inferior']
PREDICTOR_FIELD = 'Event'

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 1
survival_test_ds = df_to_dataset(test_data, PREDICTOR_FIELD, batch_size=batch_size)

# Create categorical features
vocab_file_list = build_vocab_files(test_data, categorical_col_list)
tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list)

# Create feature layer
claim_feature_columns = tf_cat_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

with open('model.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
survival_model = pickle.load(open('model.pkl', 'rb'))

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

print(classification_report(survival_yhat, pred_test))
print(roc_auc_score(survival_yhat, pred_test))
