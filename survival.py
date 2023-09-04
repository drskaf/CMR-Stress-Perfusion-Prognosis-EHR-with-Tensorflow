import numpy as np
import pandas as pd
import lifelines
from matplotlib import pyplot as plt
from lifelines.statistics import KaplanMeierFitter
from lifelines import LogNormalFitter
from lifelines.statistics import logrank_test
from lifelines import WeibullFitter
from lifelines import WeibullAFTFitter
from lifelines.plotting import qq_plot   
from lifelines import CoxPHFitter
from lifelines.utils import find_best_parametric_model

survival_df = pd.read_csv('final.csv')
survival_df['duration'] = [(x.split(' ')[0]) for x in survival_df['Duration']]
survival_df['duration'] = pd.to_numeric(survival_df["duration"], downcast="float")
survival_dfe = survival_df[survival_df['Event']==1]
print(survival_dfe['duration'].min())
print(survival_dfe['duration'].max())

lad = (survival_df['LAD_perf']==1)
lcx = (survival_df['LCx_perf']==1)
rca = (survival_df['RCA_perf']==1)

ax = plt.subplot(111)

lad_km = LogNormalFitter()
lcx_km = LogNormalFitter()
rca_km = LogNormalFitter()

lad_km.fit(durations=survival_df[lad]['duration'],
               event_observed=survival_df[lad]['Event'], label="LAD ischaemia")
lad_km.plot_survival_function(ax=ax, ci_show=False)
lcx_km.fit(durations=survival_df[lcx]['duration'],
               event_observed=survival_df[lcx]['Event'], label="LCx ischaemia")
lcx_km.plot_survival_function(ax=ax,ci_show=False)
rca_km.fit(durations=survival_df[rca]['duration'],
               event_observed=survival_df[rca]['Event'], label="RCA ischaemia")
rca_km.plot_survival_function(ax=ax,ci_show=False)

total = LogNormalFitter()
total.fit(durations=survival_dfe['duration'],
               event_observed=survival_dfe['Event'], label="total ischaemia")
print("The median survival duration (days) of patients with positive ischaemia: ", total.median_survival_time_)

# Print out the median survival duration of each group
print("The median survival duration (days) of patients with LAD ischaemia: ", lad_km.median_survival_time_)
print("The median survival duration (days) of patients with LCx ischaemia: ", lcx_km.median_survival_time_)
print("The median survival duration (days) of patients with RCA ischaemia: ", rca_km.median_survival_time_)
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.text(1000, 0.85, "Logrank p value=0.643", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.show()

patient_results = logrank_test(durations_A = survival_df[lad]['duration'],
                               durations_B = survival_df[lcx]['duration'],
                               duration_C = survival_df[rca]['duration'],
                               event_observed_A = survival_df[lad]['Event'],
                               event_observed_B = survival_df[lcx]['Event'],
                               event_observed_C = survival_df[rca]['Event'])

# Print out the p-value of log-rank test results
print('Logrank p value for coronary ischaemia:\n{}'.format(patient_results.p_value))

# Survival based on perfusion

pos = (survival_df['Positive_perf']==1)
neg = (survival_df['Positive_perf']==0)
ax = plt.subplot(111)
pos_km = LogNormalFitter()
neg_km = LogNormalFitter()
pos_km.fit(durations=survival_df[pos]['duration'],
               event_observed=survival_df[pos]['Event'], label="Positive ischaemia")
pos_km.plot_survival_function(ax=ax)
neg_km.fit(durations=survival_df[neg]['duration'],
               event_observed=survival_df[neg]['Event'], label="Negative ischaemia")
neg_km.plot_survival_function(ax=ax)
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.text(1000, 0.8, "Logrank p value <0.001", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.show()
patient_results = logrank_test(durations_A = survival_df[pos]['duration'],
                               durations_B = survival_df[neg]['duration'],
                               event_observed_A = survival_df[pos]['Event'],
                               event_observed_B = survival_df[neg]['Event'])
# Print out the p-value of log-rank test results
print('Log Rank test for perfusion:\n{}'.format(patient_results.p_value))

lad_lge = (survival_df['LAD_LGE']==1)
lcx_lge = (survival_df['LCx_LGE']==1)
rca_lge = (survival_df['RCA_LGE']==1)
ax = plt.subplot(111)
ladlge_km = LogNormalFitter()
lcxlge_km = LogNormalFitter()
rcalge_km = LogNormalFitter()
ladlge_km.fit(durations=survival_df[lad_lge]['duration'],
               event_observed=survival_df[lad_lge]['Event'], label="LAD LGE")
ladlge_km.plot_survival_function(ax=ax, ci_show=False)
lcxlge_km.fit(durations=survival_df[lcx_lge]['duration'],
               event_observed=survival_df[lcx_lge]['Event'], label="LCx LGE")
lcxlge_km.plot_survival_function(ax=ax, ci_show=False)
rcalge_km.fit(durations=survival_df[rca_lge]['duration'],
               event_observed=survival_df[rca_lge]['Event'], label="RCA LGE")
rcalge_km.plot_survival_function(ax=ax, ci_show=False)
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.text(1000, 0.8, "Logrank p value=0.305", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.show()
patient_results = logrank_test(durations_A = survival_df[lad_lge]['duration'],
                               durations_B = survival_df[lcx_lge]['duration'],
                               duration_C = survival_df[rca_lge]['duration'],
                               event_observed_A = survival_df[lad_lge]['Event'],
                               event_observed_B = survival_df[lcx_lge]['Event'],
                               event_observed_C = survival_df[rca_lge]['Event'])
# Print out the p-value of log-rank test results
print('Logrank p value for coronary LGE:\n{}'.format(patient_results.p_value))

poslge = (survival_df['Positive_LGE']==1)
neglge = (survival_df['Positive_LGE']==0)
ax = plt.subplot(111)
pos_lge = LogNormalFitter()
neg_lge = LogNormalFitter()

pos_lge.fit(durations=survival_df[poslge]['duration'],
               event_observed=survival_df[poslge]['Event'], label="Positive ischaemic LGE")
pos_lge.plot_survival_function(ax=ax)
neg_lge.fit(durations=survival_df[neglge]['duration'],
               event_observed=survival_df[neglge]['Event'], label="Negative ischaemic LGE")
neg_lge.plot_survival_function(ax=ax)
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.text(500, 0.8, "Logrank p value <0.001", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.show()
patient_results = logrank_test(durations_A = survival_df[poslge]['duration'],
                               durations_B = survival_df[neglge]['duration'],
                               event_observed_A = survival_df[poslge]['Event'],
                               event_observed_B = survival_df[neglge]['Event'])
# Print out the p-value of log-rank test results
print('Log Rank test for LGE:\n{}'.format(patient_results.p_value))

survival_df['Diabetes_mellitus'] = survival_df['Diabetes_mellitus_(disorder)']
survival_df['Cerebrovascular_accident'] = survival_df['Cerebrovascular_accident_(disorder)']
survival_df['Chronic_kidney_disease'] = survival_df['Chronic_kidney_disease_(disorder)']
survival_df['Heart_failure'] = survival_df['Heart_failure_(disorder)']
survival_df['Age'] = survival_df['Age_on_20.08.2021']
survival_df['Gender'] = survival_df['patient_GenderCode']
survival_df['LVEF'] = survival_df['LVEF_(%)']
survival_df['RVEF'] = survival_df['RVEF_(%)']

print(survival_df.head())

best_model, best_aic = find_best_parametric_model(event_times=survival_df['duration'], event_observed=survival_df['Event'],scoring_method='AIC')
print('Best model is: {}'.format(best_model))

# calculating hazard ratio models for all cause death
cox = CoxPHFitter()
cox.fit(df=survival_df, duration_col='duration', event_col='Event', formula= 'Age + Gender + Essential_hypertension + Dyslipidaemia + Diabetes_mellitus + Cerebrovascular_accident + Heart_failure + Chronic_kidney_disease + Smoking_history ')
print(cox.summary)
cox.baseline_hazard_.plot()
plt.xlabel('Time (Days)')
plt.ylabel('All Cause Mortality')
plt.title('Clinical Model')
plt.show()

cox.check_assumptions(survival_df, p_value_threshold=0.05)
cox.plot()
plt.show()

cox.fit(df=survival_df, duration_col='duration', event_col='Event', formula= 'Positive_perf + Positive_LGE ')
print(cox.summary)
cox.baseline_hazard_.plot()
plt.xlabel('Time (Days)')
plt.ylabel('All Cause Mortality')
plt.title('CMR Model')
plt.show()

cox.check_assumptions(survival_df, p_value_threshold=0.05)
cox.plot()
plt.show()

# calculating hazard ratio models for VT/VF
survival_df['VT_VF'] = survival_df[['Ventricular_tachycardia_(disorder)','Ventricular_fibrillation_(disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
cox.fit(df=survival_df, duration_col='duration', event_col='VT_VF', formula= 'Age + Gender + Essential_hypertension + Dyslipidaemia + Diabetes_mellitus + Cerebrovascular_accident + Heart_failure + Chronic_kidney_disease + Smoking_history')
print(cox.summary)
cox.baseline_hazard_.plot()
plt.xlabel('Time (Days)')
plt.ylabel('Ventricular Arrhythmia')
plt.title('Clinical Model')
plt.show()

cox.check_assumptions(survival_df, p_value_threshold=0.05)
cox.plot()
plt.title('Clinical Hazard Model for Ventricular Arrhythmia')
plt.show()


# Weibull model analysis for continuous variables
aft = WeibullAFTFitter()
aft.fit(df=survival_df, duration_col='duration', event_col='Event', formula= 'LVEF')
print(aft.summary)
aft.plot_partial_effects_on_outcome(covariates='LVEF', values=[15, 25, 35, 45, 55, 65, 75])
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.title('LVEF Model')
plt.show()

aft = WeibullAFTFitter()
aft.fit(df=survival_df, duration_col='duration', event_col='Event', formula= 'RVEF')
print(aft.summary)
aft.plot_partial_effects_on_outcome(covariates='RVEF', values=[15, 25, 35, 45, 55, 65, 75])
plt.xlabel('Time (Days)')
plt.ylabel('Survival')
plt.title('RVEF Model')
plt.show()

