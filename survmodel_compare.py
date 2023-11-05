from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from sklearn.svm import SVC    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from utils import patient_dataset_splitter_compare, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
import pickle
from keras.models import model_from_json, load_model
import scipy.stats
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_columns', 500)

# Load dataset
survival_df = pd.read_csv('final.csv')
survival_df['Gender'] = survival_df['patient_GenderCode'].astype('category')
survival_df['Gender'] = survival_df['Gender'].cat.codes
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)']
survival_df['Age'] = survival_df['Age_on_20.08.2021'].astype(int)
survival_df['Hypertension'] = survival_df['Essential_hypertension']
survival_df['Gender'] = survival_df['Gender']
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)']
survival_df['Smoking'] = survival_df['Smoking_history'].astype(int)
survival_df['LVEF'] = survival_df['LVEF_(%)']

# Define columns
categorical_col_list = ['Chronic_kidney_disease','Hypertension', 'Gender', 'Heart_failure', 'Smoking', 'Positive_LGE', 'Positive_perf']
numerical_col_list= ['Age', 'LVEF']
PREDICTOR_FIELD = 'Event'

for v in survival_df['Age'].values:
    mean = survival_df['Age'].describe()['mean']
    std = survival_df['Age'].describe()['std']
    v = v - mean / std

for x in survival_df['LVEF'].values:
    mean = survival_df['LVEF'].describe()['mean']
    std = survival_df['LVEF'].describe()['std']
    x = x - mean / std

def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
    return survival_df[selected_col_list]

selected_features_df = select_model_features(survival_df, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)

# Split data
d_train, d_test = patient_dataset_splitter_compare(selected_features_df, 'patient_TrustNumber')
d_train = d_train.drop(columns=['patient_TrustNumber'])
d_test = d_test.drop(columns=['patient_TrustNumber'])

x_train = d_train[categorical_col_list + numerical_col_list]
y_train = d_train[PREDICTOR_FIELD]
x_test = d_test[categorical_col_list + numerical_col_list]
y_test = d_test[PREDICTOR_FIELD]

# fit SVM model
svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(x_train, y_train)
svc_predict = svc_model.predict(x_test)
svc_preds = svc_model.predict_proba(x_test)[:,1]

print('SVM ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('SVM Accuracy score:',accuracy_score(y_test, svc_predict))
print('SVM F1 score:',f1_score(y_test, svc_predict))

# build linear regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
lr_preds = lr.predict_proba(x_test)[:,1]

print('LR ROCAUC score:',roc_auc_score(y_test, lr_predict))
print('LR Accuracy score:',accuracy_score(y_test, lr_predict))
print('LR F1 score:',f1_score(y_test, lr_predict))

fpr, tpr, _ = roc_curve(y_test, lr_preds)
auc = round(roc_auc_score(y_test, lr_preds), 2)
plt.plot(fpr, tpr, label="Linear Model, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()

# build random forest model
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)
rfc_predict = rfc.predict(x_test)
rfc_preds = rfc.predict_proba(x_test)[:,1]

print('RF ROCAUC score:',roc_auc_score(y_test, rfc_predict))
print('RF Accuracy score:',accuracy_score(y_test, rfc_predict))
print('RF F1 score:',f1_score(y_test, rfc_predict))

# test neural network model

# Load and preprocess test data
test_data = d_test

processed_df = preprocess_df(test_data, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

for v in processed_df['Age'].values:
    mean = processed_df['Age'].describe()['mean']
    std = processed_df['Age'].describe()['std']
    v = v - mean / std

for x in processed_df['LVEF'].values:
    mean = processed_df['LVEF'].describe()['mean']
    std = processed_df['LVEF'].describe()['std']
    x = x - mean / std

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

with open('fcn1.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
survival_model = pickle.load(open('fcn1.pkl', 'rb'))

# Predict with model
preds = survival_model.predict(survival_test_ds)
pred_test_cl = []
for p in preds:
    pred = np.argmax(p)
    pred_test_cl.append(pred)
print(pred_test_cl[:5])
survival_yhat = list(test_data['Event'].values)
print(survival_yhat[:5])

prob_outputs = {
    "pred": pred_test_cl,
    "actual_value": survival_yhat
}
prob_output_df = pd.DataFrame(prob_outputs)
print(prob_output_df.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl))
print('Clinical FCN ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl))
print('Clinical FCN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl))
print('Clinical FCN F1 score:',f1_score(survival_yhat, pred_test_cl))

# build XGBoost Classifier model
xgb_model = XGBClassifier(tree_method="hist",enable_categorical=True).fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)
xgb_preds = xgb_model.predict_proba(x_test)[:,1]
print('XGB ROCAUC score:',roc_auc_score(y_test, xgb_predict))
print('XGB Accuracy score:',accuracy_score(y_test, xgb_predict))
print('XGB F1 score:',f1_score(y_test, xgb_predict))

# build ensemble method
comb_model = VotingClassifier(estimators=[('XBG',xgb_model), ('LR',lr), ('RF',rfc), ('SVC',svc_model)], voting='soft')
comb_model.fit(x_train, y_train)
comb_model_pred = comb_model.predict(x_test)
comb_preds = comb_model.predict_proba(x_test)[:,1]

print('Ensemble Model ROCAUC score:',roc_auc_score(y_test, comb_model_pred))
print('Ensemble Model Accuracy score:',accuracy_score(y_test, comb_model_pred))
print('Ensemble Model F1 score:',f1_score(y_test, comb_model_pred))

fpr, tpr, _ = roc_curve(y_test, comb_preds)
auc = round(roc_auc_score(y_test, comb_preds), 2)
plt.plot(fpr, tpr, label="Non-Linear Model, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()

# plot AUC

fpr, tpr, _ = roc_curve(y_test, lr_preds)
auc = round(roc_auc_score(y_test, lr_preds), 2)
plt.plot(fpr, tpr, label="Multivariate Regression, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, rfc_preds)
auc = round(roc_auc_score(y_test, rfc_preds), 2)
plt.plot(fpr, tpr, label="Random Forest, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, xgb_preds)
auc = round(roc_auc_score(y_test, xgb_preds), 2)
plt.plot(fpr, tpr, label="XGBoost Classifier, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, svc_preds)
auc = round(roc_auc_score(y_test, svc_preds), 2)
plt.plot(fpr,tpr,label="SVM, AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds[:,1])
auc = round(roc_auc_score(survival_yhat, preds[:,1]), 2)
plt.plot(fpr, tpr, label="Fully Connected Network, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, comb_preds)
auc = round(roc_auc_score(y_test, comb_preds), 2)
plt.plot(fpr, tpr, label="Ensemble Classifier, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Survival Models Comparison')
plt.show()


# DeLong test

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

print('DeLong test for non-linear and linear predictions:', delong_roc_test(y_test, svc_predict, lr_predict))

