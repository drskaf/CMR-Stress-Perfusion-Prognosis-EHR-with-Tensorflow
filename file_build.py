from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae
import os
from itertools import chain

main_df = pd.read_csv('Final list.csv')
df1 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotation corrected.csv')
df2 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 0902_1801.csv')
df3 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 1802_2701.csv')
df4 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 2701_3501.csv')
df5 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 3501_4201.csv')
df6 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 4201_5001.csv')
df7 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 5001_6001.csv')
df8 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 6001_6501.csv')
df9 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations_01_901.csv')

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
df = pd.concat(frames)

cvs_labels = np.unique(list(chain(*df['nlp.pretty_name'].map(lambda x: x.split('|')).tolist())))
cvs_labels = [x for x in cvs_labels if len(x)>0]
print('CVS Labels ({}): {}'.format(len(cvs_labels), cvs_labels))
for c_label in cvs_labels:
    if len(c_label)>1: # leave out empty labels
        df[c_label] = df['nlp.pretty_name'].map(lambda finding: 1.0 if c_label in finding else 0)
df = df.drop(columns=['nlp.cui', 'nlp.pretty_name','nlp.source_value','meta.document_TouchedWhen'])
df = df.groupby('meta.patient_TrustNumber').agg(lambda x: np.max(x))

print(df.head())
print(df.tail())
print(len(df))

df.to_csv('CVS_data.csv')
