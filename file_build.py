from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import pandas as pd
from itertools import chain
from datetime import datetime, date   

main_df = pd.read_csv('AI Perfusion Data.csv')
df1 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotation corrected.csv')
df2 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 0902_1801.csv')
df3 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 1802_2701.csv')
df4 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 2701_3501.csv')
df5 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 3501_4201.csv')
df6 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 4201_5001.csv')
df7 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 5001_6001.csv')
df8 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations 6001_6501.csv')
df9 = pd.read_csv('CV_MRI_Comorbid/Cardiovascular MRI-annotations_01_901.csv')
c_smokedf = pd.read_csv('CV_MRI_Comorbid/current smoker.csv')
ex_smokedf = pd.read_csv('CV_MRI_Comorbid/Ex smoker.csv')
c_smokedf = c_smokedf.set_index('patient_TrustNumber')
ex_smokedf = ex_smokedf.set_index('patient_TrustNumber')
c_smokedf['Smoker'] = 1
ex_smokedf['Ex'] = 1
c_smokedf = c_smokedf.drop(columns=['document_MeasurementDate', 'Unnamed: 0'])
c_smokedf = c_smokedf.groupby('patient_TrustNumber').agg(lambda x:np.max(x))
ex_smokedf = ex_smokedf.drop(columns=['document_MeasurementDate', 'Unnamed: 0'])
ex_smokedf = ex_smokedf.groupby('patient_TrustNumber').agg(lambda x:np.max(x))

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
df = pd.concat(frames)
df['patient_TrustNumber'] = df['meta.patient_TrustNumber']

cvs_labels = np.unique(list(chain(*df['nlp.pretty_name'].map(lambda x: x.split('|')).tolist())))
cvs_labels = [x for x in cvs_labels if len(x)>0]
print('CVS Labels ({}): {}'.format(len(cvs_labels), cvs_labels))

for c_label in cvs_labels:
    if len(c_label)>1: # leave out empty labels
        df[c_label] = df['nlp.pretty_name'].map(lambda finding: 1.0 if c_label in finding else 0)
df = df.drop(columns=['nlp.cui', 'nlp.pretty_name','nlp.source_value','meta.document_TouchedWhen'])
df = df.groupby('meta.patient_TrustNumber').agg(lambda x: np.max(x))
print(df.head())

df.set_index('patient_TrustNumber')
df = df.join(c_smokedf, on='patient_TrustNumber').fillna(0)
df = df.join(ex_smokedf, on='patient_TrustNumber').fillna(0)

main_df = main_df.drop(columns=['ID','Patient_name','Accession.number','First_Name','Surname','patient_ReligionCode','duplicated','Num_Names','patient_Id','patient_MaritalStatusCode','patient_ReligionCode','Angio'])
main_df = main_df.set_index('patient_TrustNumber')
merge_df = main_df.join(df).fillna(0)

merge_df['Smoking history'] = merge_df[['Smoker','Ex']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['Ex smoker'] = merge_df.apply(lambda x : 0 if x['Smoker'] == x['Ex'] and x['Smoker'] ==1  else x['Ex'], axis=1)
merge_df= merge_df.drop(columns=['Ex'])

merge_df['Essential hypertension'] = merge_df[['Essential hypertension (disorder)','Hypertensive disorder, systemic arterial (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)

merge_df['Dyslipidaemia'] = merge_df[['Dyslipidemia (disorder)','Hypercholesterolemia (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['t1'] = pd.to_datetime(merge_df['patient_DeceasedDtm'])
merge_df['t2'] = pd.to_datetime(merge_df['Date_of_CMR'], dayfirst=True)
merge_df['Duration'] = merge_df['t1'] - merge_df['t2']

print(merge_df.head())
print(len(merge_df))

merge_df.to_csv('final.csv')

merge_df['LAD_perf'] = merge_df[
    ['p_basal anterior', 'p_basal anteroseptum', 'p_mid anterior', 'p_mid anteroseptum', 'p_apical anterior',
     'p_apical septum']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['LCx_perf'] = merge_df[
    ['p_basal inferolateral', 'p_basal anterolateral', 'p_mid inferolateral', 'p_mid anterolateral',
     'p_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['RCA_perf'] = merge_df[
    ['p_basal inferoseptum', 'p_basal inferior', 'p_mid inferoseptum', 'p_mid inferior', 'p_apical inferior']].apply(
    lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['Positive_perf'] = merge_df[['LAD_perf', 'LCx_perf', 'RCA_perf']].apply(lambda x: '{}'.format(np.max(x)),
                                                                                 axis=1)

merge_df['LAD_LGE'] = merge_df[
    ['LGE_basal anterior', 'LGE_basal anteroseptum', 'LGE_mid anterior', 'LGE_mid anteroseptum', 'LGE_apical anterior',
     'LGE_apical septum', 'True_apex']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['LCx_LGE'] = merge_df[
    ['LGE_basal inferolateral', 'LGE_basal anterolateral', 'LGE_mid inferolateral', 'LGE_mid anterolateral',
     'LGE_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['RCA_LGE'] = merge_df[
    ['LGE_basal inferoseptum', 'LGE_basal inferior', 'LGE_mid inferoseptum', 'LGE_mid inferior',
     'LGE_apical inferior']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
merge_df['Positive_LGE'] = merge_df[['LAD_LGE', 'LCx_LGE', 'RCA_LGE']].apply(lambda x: '{}'.format(np.max(x)), axis=1)

merge_df.columns = merge_df.columns.str.replace(' ','_')

print(merge_df.head())


merge_df.to_csv('final.csv')
    
