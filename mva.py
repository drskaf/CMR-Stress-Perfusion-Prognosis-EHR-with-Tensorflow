import numpy as np
import pandas as pd
import lifelines
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols, logit
import seaborn as sns

survival_df = pd.read_csv('final.csv')

survival_df['Diabetes_mellitus'] = survival_df['Diabetes_mellitus_(disorder)'].astype(int)
survival_df['Cerebrovascular_accident'] = survival_df['Cerebrovascular_accident_(disorder)'].astype(int)
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)'].astype(int)
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)'].astype(int)
survival_df['Age'] = survival_df['Age_on_20.08.2021'].astype(int)
survival_df['Gender'] = survival_df['patient_GenderCode'].astype('category')
survival_df['Gender'] = survival_df['Gender'].cat.codes
survival_df['Essential_hypertension'] = survival_df['Essential_hypertension'].astype(int)
survival_df['Dyslipidaemia'] = survival_df['Dyslipidaemia'].astype(int)

survival_df['VT_VF'] = survival_df[['Ventricular_tachycardia_(disorder)','Ventricular_fibrillation_(disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
survival_df['VT_VF'] = survival_df['VT_VF'].astype(float)

# Regression for continuous variables

survival_df = survival_df[['VT_VF', 'Age', 'Gender', 'Essential_hypertension', 'Dyslipidaemia', 'Diabetes_mellitus', 'Cerebrovascular_accident', 'Heart_failure', 'Chronic_kidney_disease', 'Smoking_history', 'Positive_LGE', 'Positive_perf']]

ml_reg = logit('VT_VF ~ Age + Gender + Essential_hypertension + Dyslipidaemia + Diabetes_mellitus + Cerebrovascular_accident + Heart_failure + Chronic_kidney_disease + Smoking_history + Positive_LGE + Positive_perf', data=survival_df).fit()
print(ml_reg.summary())

survival_df = survival_df.rename(columns={'Diabetes_mellitus':'Diabetes mellitus', 'Cerebrovascular_accident': 'Cerebroascular accident',
'Chronic_kidney_disease': 'Chronic kidney disease', 'Heart_failure': 'Heart failure', 'Essential_hypertension': 'Essential hypertension',
                                          'Smoking_history': 'Smoking history', 'Positive_LGE': 'Positive ischaemic LGE',
                                          'Positive_perf': 'Positive perfusion'})

survival_df = survival_df.drop(columns=['VT_VF'])
ax = sns.heatmap(survival_df.corr(),annot=True, fmt='.2f')
plt.show()
