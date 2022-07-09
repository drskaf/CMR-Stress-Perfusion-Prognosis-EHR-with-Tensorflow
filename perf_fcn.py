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
from utils import create_tf_categorical_feature_cols, patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification

pd.set_option('display.max_columns', 500)

# Load dataset
survival_df = pd.read_csv('final.csv')
survival_df['Gender'] = survival_df['patient_GenderCode'].astype('category')
survival_df['Gender'] = survival_df['Gender'].cat.codes
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)'].astype(str)
survival_df['Age'] = survival_df['Age_on_20.08.2021'].astype(int)
survival_df['Hypertension'] = survival_df['Essential_hypertension'].astype(str)
survival_df['Gender'] = survival_df['Gender'].astype(str)
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)'].astype(str)

# Define columns
categorical_col_list = ['p_basal_anterior', 'p_basal_anteroseptum', 'p_mid_anterior', 'p_mid_anteroseptum', 'p_apical_anterior',
     'p_apical_septum','p_basal_inferolateral', 'p_basal_anterolateral', 'p_mid_inferolateral', 'p_mid_anterolateral',
     'p_apical_lateral','p_basal_inferoseptum', 'p_basal_inferior', 'p_mid_inferoseptum', 'p_mid_inferior', 'p_apical_inferior']
PREDICTOR_FIELD = 'Event'

# preprocess the data
def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
    return survival_df[selected_col_list]


selected_features_df = select_model_features(survival_df, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)
processed_df = preprocess_df(selected_features_df, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

# Split data
d_train, d_val, d_test = patient_dataset_splitter(survival_df, 'patient_TrustNumber')
assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)
print("Test passed for number of total rows equal!")
d_test.to_csv('test_data.csv')

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
tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list)

# Test a batch
test_cat_var1 = tf_cat_col_list[0]
print("Example categorical field:\n{}".format(test_cat_var1))
demo(test_cat_var1, survival_batch)

import random
random.seed(123)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(1234)

# Building neural network
claim_feature_columns = tf_cat_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

def scheduler(epoch, lr):
    if epoch<10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
     
def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(175, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

checkpoint_path = "training/ann.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=False,
                                                 verbose=1)

def build_survival_model(train_ds, val_ds,  feature_layer,  epochs=5, loss_metric='mse'):
    model = build_sequential_model(feature_layer)
    model.compile(optimizer='RMSprop', loss=loss_metric, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(train_ds, validation_data=val_ds,
                        callbacks=[cp_callback,early_stop],
                        epochs=epochs)
    return model, history

survival_model, history = build_survival_model(survival_train_ds, survival_val_ds,  claim_feature_layer,  epochs=20)

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

pickle.dump(survival_model, open('model.pkl', 'wb'))


