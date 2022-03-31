from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import pandas as pd
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
df['patient_TrustNumber'] = df['meta.patient_TrustNumber']
df = df.groupby('meta.patient_TrustNumber').agg(lambda x: np.max(x))
print(df.head())

df.set_index('patient_TrustNumber')

main_df = main_df.drop(columns=['ID','Patient_name','Accession.number','First_Name','Surname','patient_ReligionCode','duplicated','M','CVM','Num_Names','patient_Id','patient_MaritalStatusCode','patient_ReligionCode'])
main_df = main_df.set_index('patient_TrustNumber')

merge_df = main_df.join(df).fillna(0)
merge_df['Essential hypertension'] = merge_df[['Essential hypertension (disorder)','Hypertensive disorder, systemic arterial (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)


print(merge_df.head())
print(len(merge_df))

merge_df.to_csv('final.csv')

survival = pd.read_csv('Survival.csv')
survival = survival.drop(columns=['ID','Patient_name','Accession.number','First_Name','Surname','patient_ReligionCode','duplicated','M','CVM','Num_Names','patient_Id','patient_MaritalStatusCode','patient_ReligionCode'])
survival = survival.set_index('patient_TrustNumber')
survival_df = survival.join(df).fillna(0)
print(len(survival_df))

survival_df['t1'] = pd.to_datetime(survival_df['patient_DeceasedDtm'])
survival_df['t2'] = pd.to_datetime(survival_df['Date_of_CMR'])
survival_df['Duration'] = survival_df['t1'] - survival_df['t2']

survival_df['LAD_perf'] = survival_df[['p_basal anterior','p_basal anteroseptum','p_mid anterior','p_mid anteroseptum','p_apical anterior','p_apical septum']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['LCx_perf'] = survival_df[['p_basal inferolateral','p_basal anterolateral','p_mid inferolateral','p_mid anterolateral','p_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['RCA_perf'] = survival_df[['p_basal inferoseptum','p_basal inferior','p_mid inferoseptum','p_mid inferior','p_apical inferior']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['Positive_perf'] = survival_df[['LAD_perf','LCx_perf','RCA_perf']].apply(lambda x: '{}'.format(np.max(x)), axis=1)

survival_df['LAD_LGE'] = survival_df[['LGE_basal anterior','LGE_basal anteroseptum','LGE_mid anterior','LGE_mid anteroseptum','LGE_apical anterior','LGE_apical septum']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['LCx_LGE'] = survival_df[['LGE_basal inferolateral','LGE_basal anterolateral','LGE_mid inferolateral','LGE_mid anterolateral','LGE_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['RCA_LGE'] = survival_df[['LGE_basal inferoseptum','LGE_basal inferior','LGE_mid inferoseptum','LGE_mid inferior','LGE_apical inferior']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['Positive_LGE'] = survival_df[['LAD_LGE','LCx_LGE','RCA_LGE']].apply(lambda x: '{}'.format(np.max(x)), axis=1)

survival_df['Essential hypertension'] = survival_df[['Essential hypertension (disorder)','Hypertensive disorder, systemic arterial (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['Dyslipidaemia'] = survival_df[['Dyslipidemia (disorder)','Hypercholesterolemia (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)

print(survival_df.head())
survival_df.to_csv('survival_final.csv')

mini_survival = pd.read_csv('Survival_mini.csv')
mini_survival = mini_survival.drop(columns=['ID','Patient_name','Accession.number','First_Name','Surname','patient_ReligionCode','duplicated','M','CVM','Num_Names','patient_Id','patient_MaritalStatusCode','patient_ReligionCode'])
mini_survival = mini_survival.set_index('patient_TrustNumber')
mini_survival = mini_survival.join(df).fillna(0)
mini_survival['Essential hypertension'] = mini_survival[['Essential hypertension (disorder)','Hypertensive disorder, systemic arterial (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['Dyslipidaemia'] = mini_survival[['Dyslipidemia (disorder)','Hypercholesterolemia (disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['LAD_perf'] = mini_survival[
    ['p_basal anterior', 'p_basal anteroseptum', 'p_mid anterior', 'p_mid anteroseptum', 'p_apical anterior',
     'p_apical septum']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['LCx_perf'] = mini_survival[
    ['p_basal inferolateral', 'p_basal anterolateral', 'p_mid inferolateral', 'p_mid anterolateral',
     'p_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['RCA_perf'] = mini_survival[
    ['p_basal inferoseptum', 'p_basal inferior', 'p_mid inferoseptum', 'p_mid inferior', 'p_apical inferior']].apply(
    lambda x: '{}'.format(np.max(x)), axis=1)


mini_survival['LAD_LGE'] = mini_survival[
    ['LGE_basal anterior', 'LGE_basal anteroseptum', 'LGE_mid anterior', 'LGE_mid anteroseptum', 'LGE_apical anterior',
     'LGE_apical septum']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['LCx_LGE'] = mini_survival[
    ['LGE_basal inferolateral', 'LGE_basal anterolateral', 'LGE_mid inferolateral', 'LGE_mid anterolateral',
     'LGE_apical lateral']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival['RCA_LGE'] = mini_survival[
    ['LGE_basal inferoseptum', 'LGE_basal inferior', 'LGE_mid inferoseptum', 'LGE_mid inferior',
     'LGE_apical inferior']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
mini_survival.to_csv('survivalm.csv')  
