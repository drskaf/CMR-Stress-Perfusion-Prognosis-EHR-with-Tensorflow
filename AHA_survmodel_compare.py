from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from student_utils import patient_dataset_splitter
from utils import build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
from plot_metric.functions import BinaryClassification
import pickle
from keras.models import model_from_json, load_model

pd.set_option('display.max_columns', 500)

# Load dataset
survival_df = pd.read_csv('final.csv')

# Define columns
categorical_col_list = ['p_basal_anterior', 'p_basal_anteroseptum', 'p_mid_anterior', 'p_mid_anteroseptum', 'p_apical_anterior',
     'p_apical_septum', 'p_basal_inferolateral', 'p_basal_anterolateral', 'p_mid_inferolateral', 'p_mid_anterolateral',
     'p_apical_lateral','p_basal_inferoseptum', 'p_basal_inferior', 'p_mid_inferoseptum', 'p_mid_inferior', 'p_apical_inferior' ]
PREDICTOR_FIELD = 'Event'

def select_model_features(df, categorical_col_list, PREDICTOR_FIELD, grouping_key='patient_TrustNumber'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list
    return survival_df[selected_col_list]

selected_features_df = select_model_features(survival_df, categorical_col_list,
                                             PREDICTOR_FIELD)

# Split data
d_train, d_val, d_test = patient_dataset_splitter(selected_features_df, 'patient_TrustNumber')
d_train = d_train.drop(columns=['patient_TrustNumber'])
d_val = d_val.drop(columns=['patient_TrustNumber'])
d_test = d_test.drop(columns=['patient_TrustNumber'])

x_train = d_train[categorical_col_list]
y_train = d_train[PREDICTOR_FIELD]
x_test = d_test[categorical_col_list]
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

# test neural network model

# Load and preprocess test data
testaha_data = pd.read_csv('testaha_data.csv')
def preprocess_df(df, categorical_col_list, predictor, categorical_impute_value='nan'):
    df[predictor] = df[predictor].astype(int)
    for c in categorical_col_list:
        df[c] = df[c].astype(str)
    return df

processed_df = preprocess_df(testaha_data, categorical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan' )

# Convert dataset from Pandas dataframes to TF dataset
batch_size = 1
survival_test_ds = df_to_dataset(processed_df, PREDICTOR_FIELD, batch_size=batch_size)

# Create categorical features
vocab_file_list = build_vocab_files(testaha_data, categorical_col_list)
from student_utils import create_tf_categorical_feature_cols
tf_cat_col_list = create_tf_categorical_feature_cols(categorical_col_list)

# Create feature layer
claim_feature_columns = tf_cat_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)

with open('fcn_a.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
survival_model = pickle.load(open('fcn_a.pkl', 'rb'))

# Predict with model
preds = survival_model.predict(survival_test_ds)
pred_test = []
for p in preds:
    pred = np.argmax(p)
    pred_test.append(pred)
print(pred_test[:5])
survival_yhat = list(testaha_data['Event'].values)
print(survival_yhat[:5])

prob_outputs = {
    "pred": pred_test,
    "actual_value": survival_yhat
}
prob_output_df = pd.DataFrame(prob_outputs)
print(prob_output_df.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test))
print('FCN ROCAUC score:',roc_auc_score(survival_yhat, pred_test))
print('FCN Accuracy score:',accuracy_score(survival_yhat, pred_test))
print('FCN F1 score:',f1_score(survival_yhat, pred_test))

# build XGBoost Classifier model
xgb_model = XGBClassifier(tree_method="hist",enable_categorical=True).fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)
print('XGB ROCAUC score:',roc_auc_score(y_test, xgb_predict))
print('XGB Accuracy score:',accuracy_score(y_test, xgb_predict))
print('XGB F1 score:',f1_score(y_test, xgb_predict))

# build ensemble method
comb_model = VotingClassifier(estimators=[('XBG',xgb_model), ('LR',lr), ('RF',rfc), ('SVC',svc_model)], voting='hard')
comb_model.fit(x_train, y_train)
comb_model_pred = comb_model.predict(x_test)

print('Ensemble Model ROCAUC score:',roc_auc_score(y_test, comb_model_pred))
print('Ensemble Model Accuracy score:',accuracy_score(y_test, comb_model_pred))
print('Ensemble Model F1 score:',f1_score(y_test, comb_model_pred))

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
#fpr, tpr, _ = roc_curve(survival_yhat, pred_test)
#auc = round(roc_auc_score(survival_yhat, pred_test), 2)
#plt.plot(fpr, tpr, label="Fully Connected Network, AUC="+str(auc))
fpr, tpr, _ = roc_curve(y_test, comb_model_pred)
auc = round(roc_auc_score(y_test, comb_model_pred), 2)
plt.plot(fpr, tpr, label="Ensemble Classifier Model, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Survival AHA Model Comparison')
plt.show()



