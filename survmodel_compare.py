from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
from utils import patient_dataset_splitter, simple_patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
import scipy.stats
from scipy import interp


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
categorical_col_list_clinical = ['Chronic_kidney_disease','Hypertension', 'Gender', 'Heart_failure', 'Smoking']
numerical_col_list= ['Age', 'LVEF']
ID = ['patient_TrustNumber']
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
    return df[selected_col_list]

selected_features_df = select_model_features(survival_df, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)
x = survival_df.loc[:, categorical_col_list + numerical_col_list]
y = survival_df.loc[:, PREDICTOR_FIELD]

# Split data
d_train, d_test = patient_dataset_splitter(selected_features_df, 'patient_TrustNumber')
d_train = d_train.drop(columns=['patient_TrustNumber'])
d_test = d_test.drop(columns=['patient_TrustNumber'])

x_train = d_train[categorical_col_list + numerical_col_list]
x_train_cli = d_train[categorical_col_list_clinical + numerical_col_list]
y_train = d_train[PREDICTOR_FIELD]
x_test = d_test[categorical_col_list + numerical_col_list]
x_test_cli = d_test[categorical_col_list_clinical + numerical_col_list]
y_test = d_test[PREDICTOR_FIELD]

# fit SVM model
svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(x_train, y_train)
svc_predict = svc_model.predict(x_test)
svc_preds = svc_model.predict_proba(x_test)[:,1]
cm = confusion_matrix(y_test, svc_predict, labels=svc_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc_model.classes_)
disp.plot()
plt.show()

svc_model_cli = SVC(class_weight='balanced', probability=True)

svc_model_cli.fit(x_train_cli, y_train)
svc_predict_cli = svc_model_cli.predict(x_test_cli)
svc_preds_cli = svc_model_cli.predict_proba(x_test_cli)[:,1]
n_splits = 6

kf =StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=42)

svc_score = cross_val_score(svc_model_cli, x_train, y_train, cv=kf)
print(np.average(svc_score))

print('SVM ROCAUC score:',roc_auc_score(y_test, svc_preds))
print('SVM Accuracy score:',accuracy_score(y_test, svc_predict))
print('SVM F1 score:',f1_score(y_test, svc_predict))
print('Precision:', precision_score(y_test, svc_predict))
print('Recall:', recall_score(y_test, svc_predict))

# K fold experiment

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6))
i = 1
for train, test in kf.split(x, y):
    prediction = svc_model.fit(x.iloc[train], y.iloc[train]).predict_proba(x.iloc[test])
    fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i + 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)
plt.show()


# build linear regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
lr_preds = lr.predict_proba(x_test)[:,1]

cm = confusion_matrix(y_test, lr_predict, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot()
plt.show()

lr_cli = LogisticRegression()
lr_cli.fit(x_train_cli, y_train)

lr_predict_cli = lr_cli.predict(x_test_cli)
lr_preds_cli = lr_cli.predict_proba(x_test_cli)[:,1]

lr_score = cross_val_score(lr_cli, x_train, y_train, cv=kf)
print(np.average(lr_score))

print('LR ROCAUC score:',roc_auc_score(y_test, lr_preds))
print('LR Accuracy score:',accuracy_score(y_test, lr_predict))
print('LR F1 score:',f1_score(y_test, lr_predict))
print('Precision:', precision_score(y_test, lr_predict))
print('Recall:', recall_score(y_test, lr_predict))

fpr, tpr, _ = roc_curve(y_test, lr_preds_cli)
auc = round(roc_auc_score(y_test, lr_preds_cli), 2)
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

cm = confusion_matrix(y_test, rfc_predict, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
disp.plot()
plt.show()

rfc_cli = RandomForestClassifier()

rfc_cli.fit(x_train_cli, y_train)
rfc_predict_cli = rfc_cli.predict(x_test_cli)
rfc_preds_cli = rfc_cli.predict_proba(x_test_cli)[:,1]

rfc_score = cross_val_score(rfc_cli, x_train, y_train, cv=kf)
print(np.average(rfc_score))

print('RF ROCAUC score:',roc_auc_score(y_test, rfc_preds))
print('RF Accuracy score:',accuracy_score(y_test, rfc_predict))
print('RF F1 score:',f1_score(y_test, rfc_predict))
print('Precision:', precision_score(y_test, rfc_predict))
print('Recall:', recall_score(y_test, rfc_predict))

# build XGBoost Classifier model
xgb_model = XGBClassifier(tree_method="hist",enable_categorical=True).fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)
xgb_preds = xgb_model.predict_proba(x_test)[:,1]

cm = confusion_matrix(y_test, xgb_predict, labels=xgb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_model.classes_)
disp.plot()
plt.show()

xgb_model_cli = XGBClassifier(tree_method="hist",enable_categorical=True).fit(x_train_cli, y_train)

xgb_predict_cli = xgb_model_cli.predict(x_test_cli)
xgb_preds_cli = xgb_model_cli.predict_proba(x_test_cli)[:,1]

xgb_score = cross_val_score(xgb_model_cli, x_train, y_train, cv=kf)
print(np.average(xgb_score))

print('XGB ROCAUC score:',roc_auc_score(y_test, xgb_preds))
print('XGB Accuracy score:',accuracy_score(y_test, xgb_predict))
print('XGB F1 score:',f1_score(y_test, xgb_predict))
print('Precision:', precision_score(y_test, xgb_predict))
print('Recall:', recall_score(y_test, xgb_predict))

# build ensemble method
comb_model = VotingClassifier(estimators=[('XBG',xgb_model), ('LR',lr), ('RF',rfc), ('SVC',svc_model)], voting='soft')
comb_model.fit(x_train, y_train)
comb_model_pred = comb_model.predict(x_test)
comb_preds = comb_model.predict_proba(x_test)[:,1]

cm = confusion_matrix(y_test, comb_model_pred, labels=comb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=comb_model.classes_)
disp.plot()
plt.show()

comb_model_cli = VotingClassifier(estimators=[('XBG',xgb_model_cli), ('LR',lr_cli), ('RF',rfc_cli), ('SVC',svc_model_cli)], voting='soft')
comb_model_cli.fit(x_train_cli, y_train)
comb_model_pred_cli = comb_model_cli.predict(x_test_cli)
comb_preds_cli = comb_model_cli.predict_proba(x_test_cli)[:,1]

comb_score = cross_val_score(comb_model_cli, x_train, y_train, cv=kf)
print(np.average(comb_score))

print('Ensemble Model ROCAUC score:',roc_auc_score(y_test, comb_preds))
print('Ensemble Model Accuracy score:',accuracy_score(y_test, comb_model_pred))
print('Ensemble Model F1 score:',f1_score(y_test, comb_model_pred))
print('Precision:', precision_score(y_test, comb_model_pred))
print('Recall:', recall_score(y_test, comb_model_pred))

fpr, tpr, _ = roc_curve(y_test, comb_preds_cli)
auc = round(roc_auc_score(y_test, comb_preds_cli), 2)
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
#fpr, tpr, _ = roc_curve(survival_yhat, preds[:,1])
#auc = round(roc_auc_score(survival_yhat, preds[:,1]), 2)
#plt.plot(fpr, tpr, label="Fully Connected Network, AUC="+str(auc))
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


