from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae
from utils import build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification

pd.set_option('display.max_columns', 500)

# Load dataset
survival_df = pd.read_csv('final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_df['p_basal_anterior'] = survival_df['p_basal_anterior'].astype(str)
survival_df['p_basal_anteroseptum'] = survival_df['p_basal_anteroseptum'].astype(str)
survival_df['p_mid_anterior'] = survival_df['p_mid_anterior'].astype(str)
survival_df['p_mid_anteroseptum'] = survival_df['p_mid_anteroseptum'].astype(str)
survival_df['p_apical_anterior'] = survival_df['p_apical_anterior'].astype(str)
survival_df['p_apical_septum'] = survival_df['p_apical_septum'].astype(str)
survival_df['p_basal_inferolateral'] = survival_df['p_basal_inferolateral'].astype(str)
survival_df['p_basal_anterolateral'] = survival_df['p_basal_anterolateral'].astype(str)
survival_df['p_mid_inferolateral'] = survival_df['p_mid_inferolateral'].astype(str)
survival_df['p_mid_anterolateral'] = survival_df['p_mid_anterolateral'].astype(str)
survival_df['p_apical_lateral'] = survival_df['p_apical_lateral'].astype(str)
survival_df['p_basal_inferoseptum'] = survival_df['p_basal_inferoseptum'].astype(str)
survival_df['p_basal_inferior'] = survival_df['p_basal_inferior'].astype(str)
survival_df['p_mid_inferoseptum'] = survival_df['p_mid_inferoseptum'].astype(str)
survival_df['p_mid_inferior'] = survival_df['p_mid_inferior'].astype(str)
survival_df['p_apical_inferior'] = survival_df['p_apical_inferior'].astype(str)

# Define columns
categorical_col_list = ['p_basal_anterior', 'p_basal_anteroseptum', 'p_mid_anterior', 'p_mid_anteroseptum', 'p_apical_anterior',
     'p_apical_septum','p_basal_inferolateral', 'p_basal_anterolateral', 'p_mid_inferolateral', 'p_mid_anterolateral',
     'p_apical_lateral','p_basal_inferoseptum', 'p_basal_inferior', 'p_mid_inferoseptum', 'p_mid_inferior', 'p_apical_inferior']
numerical_col_list = ['Age_on_20.08.2021']
PREDICTOR_FIELD = 'Event'

# Split data
from student_utils import patient_dataset_splitter
d_train, d_val, d_test = patient_dataset_splitter(survival_df, 'patient_TrustNumber')
assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)
print("Test passed for number of total rows equal!")

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 10
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

import random
random.seed(123)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(1234)

# Building neural network
claim_feature_columns = tf_cat_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

optimizer = tf.keras.optimizers.RMSprop(0.000001)
def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(175, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    return model

checkpoint_path = "training/ann.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def build_survival_model(train_ds, val_ds,  feature_layer,  epochs=5, loss_metric='mse'):
    model = build_sequential_model(feature_layer)
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=loss_metric, patience=5)
    history = model.fit(train_ds, validation_data=val_ds,
                        callbacks=[cp_callback],
                        epochs=epochs)
    return model, history

survival_model, history = build_survival_model(survival_train_ds, survival_val_ds,  claim_feature_layer,  epochs=80)

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predict with model
feature_list = categorical_col_list
survival_x_tst = dict(d_test[feature_list])
survival_yhat = survival_model(survival_x_tst)
preds = survival_model.predict(survival_test_ds)

prob_outputs = {
    "pred": preds.flatten(),
    "actual_value": d_test['Event'].values
}

prob_output_df = pd.DataFrame(prob_outputs)
print(prob_output_df.head())

# Evaluate model
from student_utils import get_student_binary_prediction
binary_df = get_student_binary_prediction(prob_output_df, 'pred_mean')

def add_pred_to_test(test_df, pred_np, demo_col_list):
    for c in demo_col_list:
        test_df[c] = test_df[c].astype(str)
    test_df['score'] = pred_np
    test_df['label_value'] = test_df['Event']
    return test_df

pred_test_df = add_pred_to_test(d_test, prob_output_df, ['Positive_perf'])
print(pred_test_df.head())

from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
y_true = pred_test_df['label_value'].values
y_pred = pred_test_df['score'].values
print(classification_report(y_true, y_pred))
print(roc_auc_score(y_true, y_pred))

# Visualisation with plot_metric
bc = BinaryClassification(y_true, y_pred, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()
