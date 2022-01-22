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
survival_df = pd.read_csv('survival_final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_df['Diabetes_mellitus'] = survival_df['Diabetes_mellitus_(disorder)'].astype(str)
survival_df['Cerebrovascular_accident'] = survival_df['Cerebrovascular_accident_(disorder)'].astype(str)
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)'].astype(str)
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)'].astype(str)
survival_df['Dyslipidaemia'] = survival_df['Dyslipidaemia'].astype(str)
survival_df['Essential_hypertension'] = survival_df['Essential_hypertension'].astype(str)
survival_df['Positive_LGE'] = survival_df['Positive_LGE'].astype(str)
survival_df['Positive_perf'] = survival_df['Positive_perf'].astype(str)

# Define columns
categorical_col_list = ['Positive_perf']
PREDICTOR_FIELD = 'Event'

# Split data
from student_utils import patient_dataset_splitter
d_train, d_val, d_test = patient_dataset_splitter(survival_df, 'patient_TrustNumber')
assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)
print("Test passed for number of total rows equal!")

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 64
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

import random
random.seed(123)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(1234)

# Building neural network
claim_feature_columns = tf_cat_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

optimizer = tf.keras.optimizers.RMSprop(0.0001)
def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(175, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tfp.layers.DenseVariational(1+1, posterior_mean_field, prior_trainable),
        tfp.layers.DistributionLambda(
            lambda t:tfp.distributions.Normal(loc=t[..., :1],
                                             scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])
                                             )
        ),
    ])
    return model

def build_survival_model(train_ds, val_ds,  feature_layer,  epochs=5, loss_metric='mse'):
    model = build_sequential_model(feature_layer)
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=[loss_metric])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=loss_metric, patience=50)
    history = model.fit(train_ds, validation_data=val_ds,
                        callbacks=[early_stop],
                        epochs=epochs)
    return model, history

survival_model, history = build_survival_model(survival_train_ds, survival_val_ds,  claim_feature_layer,  epochs=400)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predict with model
feature_list = categorical_col_list
survival_x_tst = dict(d_test[feature_list])
survival_yhat = survival_model(survival_x_tst)
preds = survival_model.predict(survival_test_ds)

from student_utils import get_mean_std_from_preds
m, s = get_mean_std_from_preds(survival_yhat)

prob_outputs = {
    "pred": preds.flatten(),
    "actual_value": d_test['Event'].values,
    "pred_mean": m.numpy().flatten(),
    "pred_std": s.numpy().flatten()
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

pred_test_df = add_pred_to_test(d_test, binary_df, ['Positive_perf'])
print(pred_test_df.head())

from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
y_true = pred_test_df['label_value'].values
y_pred = pred_test_df['score'].values
print(classification_report(y_true, y_pred))
print(roc_auc_score(y_true, y_pred))

# Assess model bias
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.bias import Bias
from aequitas.fairness import Fairness

ae_subset_df = pred_test_df [['Positive_perf','score', 'label_value']]
ae_df, _ = preprocess_input_df(ae_subset_df)
g = Group()
xtab, _ = g.get_crosstabs(ae_df)
absolute_metrics = g.list_absolute_metrics(xtab)
clean_xtab = xtab.fillna(-1)
aqp = Plot()
b = Bias()

p = aqp.plot_group_metric_all(xtab, metrics=['tpr', 'ppr', 'fdr', 'fpr'], ncols=2)


# Visualisation with plot_metric
bc = BinaryClassification(y_true, y_pred, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()
