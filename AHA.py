from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import lifelines
from matplotlib import pyplot as plt
from lifelines.statistics import KaplanMeierFitter
from lifelines import CoxPHFitter
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

survival_df = pd.read_csv('survival_final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")

aha1 = (survival_df['p_basal_anterior']==1)
aha2 = (survival_df['p_basal_anteroseptum']==1)
aha3 = (survival_df['p_basal_inferoseptum']==1)
aha4 = (survival_df['p_basal_inferior']==1)
aha5 = (survival_df['p_basal_inferolateral']==1)
aha6 = (survival_df['p_basal_anterolateral']==1)
aha7 = (survival_df['p_mid_anterior']==1)
aha8 = (survival_df['p_mid_anteroseptum']==1)
aha9 = (survival_df['p_mid_inferoseptum']==1)
aha10 = (survival_df['p_mid_inferior']==1)
aha11 = (survival_df['p_mid_inferolateral']==1)
aha12 = (survival_df['p_mid_anterolateral']==1)
aha13 = (survival_df['p_apical_anterior']==1)
aha14 = (survival_df['p_apical_septum']==1)
aha15 = (survival_df['p_apical_inferior']==1)
aha16 = (survival_df['p_apical_lateral']==1)

ax = plt.subplot(111)
aha1_km = KaplanMeierFitter()
aha2_km = KaplanMeierFitter()
aha3_km = KaplanMeierFitter()
aha4_km = KaplanMeierFitter()
aha5_km = KaplanMeierFitter()
aha6_km = KaplanMeierFitter()
aha7_km = KaplanMeierFitter()
aha8_km = KaplanMeierFitter()
aha9_km = KaplanMeierFitter()
aha10_km = KaplanMeierFitter()
aha11_km = KaplanMeierFitter()
aha12_km = KaplanMeierFitter()
aha13_km = KaplanMeierFitter()
aha14_km = KaplanMeierFitter()
aha15_km = KaplanMeierFitter()
aha16_km = KaplanMeierFitter()

aha1_km.fit(durations=survival_df[aha1]['duration'],
               event_observed=survival_df[aha1]['Event'], label="AHA1 ischaemia")
aha1_km.plot_survival_function(ax=ax, ci_show=False)
aha2_km.fit(durations=survival_df[aha2]['duration'],
               event_observed=survival_df[aha2]['Event'], label="AHA2 ischaemia")
aha2_km.plot_survival_function(ax=ax, ci_show=False)
aha3_km.fit(durations=survival_df[aha3]['duration'],
               event_observed=survival_df[aha3]['Event'], label="AHA3 ischaemia")
aha3_km.plot_survival_function(ax=ax, ci_show=False)
aha4_km.fit(durations=survival_df[aha4]['duration'],
               event_observed=survival_df[aha4]['Event'], label="AHA4 ischaemia")
aha4_km.plot_survival_function(ax=ax, ci_show=False)
aha5_km.fit(durations=survival_df[aha5]['duration'],
               event_observed=survival_df[aha5]['Event'], label="AHA5 ischaemia")
aha5_km.plot_survival_function(ax=ax, ci_show=False)
aha6_km.fit(durations=survival_df[aha6]['duration'],
               event_observed=survival_df[aha6]['Event'], label="AHA6 ischaemia")
aha6_km.plot_survival_function(ax=ax, ci_show=False)
aha7_km.fit(durations=survival_df[aha7]['duration'],
               event_observed=survival_df[aha7]['Event'], label="AHA7 ischaemia")
aha7_km.plot_survival_function(ax=ax, ci_show=False)
aha8_km.fit(durations=survival_df[aha8]['duration'],
               event_observed=survival_df[aha8]['Event'], label="AHA8 ischaemia")
aha8_km.plot_survival_function(ax=ax, ci_show=False)
aha9_km.fit(durations=survival_df[aha9]['duration'],
               event_observed=survival_df[aha9]['Event'], label="AHA9 ischaemia")
aha9_km.plot_survival_function(ax=ax, ci_show=False)
aha10_km.fit(durations=survival_df[aha10]['duration'],
               event_observed=survival_df[aha10]['Event'], label="AHA10 ischaemia")
aha10_km.plot_survival_function(ax=ax, ci_show=False)
aha11_km.fit(durations=survival_df[aha11]['duration'],
               event_observed=survival_df[aha11]['Event'], label="AHA11 ischaemia")
aha11_km.plot_survival_function(ax=ax, ci_show=False)
aha12_km.fit(durations=survival_df[aha12]['duration'],
               event_observed=survival_df[aha12]['Event'], label="AHA12 ischaemia")
aha12_km.plot_survival_function(ax=ax, ci_show=False)
aha13_km.fit(durations=survival_df[aha13]['duration'],
               event_observed=survival_df[aha13]['Event'], label="AHA13 ischaemia")
aha13_km.plot_survival_function(ax=ax, ci_show=False)
aha14_km.fit(durations=survival_df[aha14]['duration'],
               event_observed=survival_df[aha14]['Event'], label="AHA14 ischaemia")
aha14_km.plot_survival_function(ax=ax, ci_show=False)
aha15_km.fit(durations=survival_df[aha15]['duration'],
               event_observed=survival_df[aha15]['Event'], label="AHA15 ischaemia")
aha15_km.plot_survival_function(ax=ax, ci_show=False)
aha16_km.fit(durations=survival_df[aha16]['duration'],
               event_observed=survival_df[aha16]['Event'], label="AHA16 ischaemia")
aha16_km.plot_survival_function(ax=ax, ci_show=False)


plt.legend(fontsize=7)
plt.show()


pd.set_option('display.max_columns', 500)

# Load dataset
survival_df = pd.read_csv('survival_final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_df['p_basal_anterior'] = survival_df['p_basal_anterior'].astype(str)
survival_df['p_basal_anteroseptum'] = survival_df['p_basal_anteroseptum'].astype(str)
survival_df['p_basal_inferoseptum'] = survival_df['p_basal_inferoseptum'].astype(str)
survival_df['p_basal_inferior'] = survival_df['p_basal_inferior'].astype(str)
survival_df['p_basal_inferolateral'] = survival_df['p_basal_inferolateral'].astype(str)
survival_df['p_basal_anterolateral'] = survival_df['p_basal_anterolateral'].astype(str)
survival_df['p_mid_anterior'] = survival_df['p_mid_anterior'].astype(str)
survival_df['p_mid_anteroseptum'] = survival_df['p_mid_anteroseptum'].astype(str)
survival_df['p_mid_inferoseptum'] = survival_df['p_mid_inferoseptum'].astype(str)
survival_df['p_mid_inferior'] = survival_df['p_mid_inferior'].astype(str)
survival_df['p_mid_inferolateral'] = survival_df['p_mid_inferolateral'].astype(str)
survival_df['p_mid_anterolateral'] = survival_df['p_mid_anterolateral'].astype(str)
survival_df['p_apical_anterior'] = survival_df['p_apical_anterior'].astype(str)
survival_df['p_apical_septum'] = survival_df['p_apical_septum'].astype(str)
survival_df['p_apical_inferior'] = survival_df['p_apical_inferior'].astype(str)
survival_df['p_apical_lateral'] = survival_df['p_apical_lateral'].astype(str)

# Define columns
categorical_col_list = ['p_basal_anterior','p_basal_anteroseptum','p_basal_inferoseptum','p_basal_inferior','p_basal_inferolateral','p_basal_anterolateral','p_mid_anterior','p_mid_anteroseptum','p_mid_inferoseptum','p_mid_inferior','p_mid_inferolateral','p_mid_anterolateral','p_apical_anterior','p_apical_septum','p_apical_inferior','p_apical_lateral']
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

pred_test_df = add_pred_to_test(d_test, binary_df, ['p_basal_anterior','p_basal_anteroseptum'])
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

ae_subset_df = pred_test_df [['p_basal_anterior','p_basal_anteroseptum','score', 'label_value']]
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
