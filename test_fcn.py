import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
from utils import patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification

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
