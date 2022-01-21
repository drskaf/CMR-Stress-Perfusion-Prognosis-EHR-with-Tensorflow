from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae

survival_df = pd.read_csv('survival_final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_df['Diabetes_mellitus'] = survival_df['Diabetes_mellitus_(disorder)']
survival_df['Cerebrovascular_accident'] = survival_df['Cerebrovascular_accident_(disorder)']
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)']
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)']

categorical_col_list = [ 'patient_GenderCode + Essential_hypertension + Dyslipidaemia + Positive_LGE + Positive_perf + Diabetes_mellitus + Cerebrovascular_accident + Heart_failure + Chronic_kidney_disease' ]
numerical_col_list = [ 'Age_on_20.08.2021' ]
PREDICTOR_FIELD = ['duration']

from student_utils import patient_dataset_splitter
d_train, d_val, d_test = patient_dataset_splitter(survival_df, 'patient_TrustNumber')
assert len(d_train) + len(d_val) + len(d_test) == len(survival_df)
print("Test passed for number of total rows equal!")
