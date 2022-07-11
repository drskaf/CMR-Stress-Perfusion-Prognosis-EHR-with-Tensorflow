from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from utils import  patient_dataset_splitter, build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
import pickle

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
categorical_col_list = ['Chronic_kidney_disease','Hypertension', 'Gender', 'Heart_failure' ]
numerical_col_list= ['Age']
PREDICTOR_FIELD = 'Event'

def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
    return survival_df[selected_col_list]

selected_features_df = select_model_features(survival_df, categorical_col_list, numerical_col_list,
                                             PREDICTOR_FIELD)
processed_df = preprocess_df(selected_features_df, categorical_col_list,
        numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)

for c in categorical_col_list:
    selected_features_df[c] = selected_features_df[c].astype(bool)

# Split data
d_train, d_val, d_test = patient_dataset_splitter(selected_features_df, 'patient_TrustNumber')
d_train = d_train.drop(columns=['patient_TrustNumber'])
d_val = d_val.drop(columns=['patient_TrustNumber'])
d_train.to_csv('train_data.csv')
d_val.to_csv('valid_data.csv')
d_test.to_csv('test_data.csv')

x_train = d_train[categorical_col_list + numerical_col_list]
y_train = d_train[PREDICTOR_FIELD]
x_test = d_test[categorical_col_list + numerical_col_list]
y_test = d_test[PREDICTOR_FIELD]

# fit SVM model
svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(x_train, y_train)
svc_predict = svc_model.predict(x_test)

print('SVM ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('SVM Accuracy score:',accuracy_score(y_test, svc_predict))
print('SVM F1 score:',f1_score(y_test, svc_predict))

# build linear regression model
lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
print('LR ROCAUC score:',roc_auc_score(y_test, lr_predict))
print('LR Accuracy score:',accuracy_score(y_test, lr_predict))
print('LR F1 score:',f1_score(y_test, lr_predict))

# build random forest model
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

rfc_predict = rfc.predict(x_test)# check performance
print('RF ROCAUC score:',roc_auc_score(y_test, rfc_predict))
print('RF Accuracy score:',accuracy_score(y_test, rfc_predict))
print('RF F1 score:',f1_score(y_test, rfc_predict))

# build XGBoost Classifier model
xgb_model = XGBClassifier().fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)
print('XGB ROCAUC score:',roc_auc_score(y_test, xgb_predict))
print('XGB Accuracy score:',accuracy_score(y_test, xgb_predict))
print('XGB F1 score:',f1_score(y_test, xgb_predict))

# plot AUC
fpr, tpr, _ = roc_curve(y_test, svc_predict)
auc = round(roc_auc_score(y_test, svc_predict), 2)
plt.plot(fpr,tpr,label="SVM Model, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, lr_predict)
auc = round(roc_auc_score(y_test, lr_predict), 2)
plt.plot(fpr, tpr, label="Linear Regression Model, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, rfc_predict)
auc = round(roc_auc_score(y_test, rfc_predict), 2)
plt.plot(fpr, tpr, label="Random Forest Model, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, xgb_predict)
auc = round(roc_auc_score(y_test, xgb_predict), 2)
plt.plot(fpr, tpr, label="XGBoost Classifier Model, AUC="+str(auc))
plt.legend()
plt.show()




