import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
sns.set_theme(style="ticks", color_codes=True)

# loading data
data = pd.read_csv('final.csv')
# encoding gender variable
data['Gender'] = data['patient_GenderCode'].astype('category')
data['Gender'] = data['Gender'].cat.codes
# concatenating columns
data['CVA'] = data[['Cerebrovascular_accident_(disorder)','Transient_ischemic_attack_(disorder)']].apply(lambda x: '{}'.format(np.max(x)), axis=1)
data['CVA'] = data['CVA'].astype('category')
data['CVA'] = data['CVA'].cat.codes
x = {'1.5T Philips':0, '1.5T Siemens':0, '1.5T':0, '3T':1, '3T Philips':1, '3T Siemens':1}
data['Field_strength'] = data['Field_strength'].map(x)
data['Field_strength'] = data['Field_strength'].astype('category')
data['Field_strength'] = data['Field_strength'].cat.codes
y = {'A':0, 'na':0, 'R': 1, 'A + R': 1}
data['Stress_agent'] = data['Stress_agent'].map(y)
data['Stress_agent'] = data['Stress_agent'].astype('category')
data['Stress_agent'] = data['Stress_agent'].cat.codes
print(data['Stress_agent'].head())

# dividing into age group
age_group1 = data[data['Age_on_20.08.2021'] <65]
age_group2 = data[(data['Age_on_20.08.2021'] >=65) & (data['Age_on_20.08.2021'] <=75)]
age_group3 = data[data['Age_on_20.08.2021'] >75]
print(len(age_group1))
print(len(age_group2))
print(len(age_group3))

# calculating events
event1 = (age_group1[age_group1['Event']==1])
event2 = (age_group2[age_group2['Event']==1])
event3 = (age_group3[age_group3['Event']==1])
print('Number of events in group 1:\n{}'.format(len(event1)))
print('Number of events in group 2:\n{}'.format(len(event2)))
print('Number of events in group 3:\n{}'.format(len(event3)))
print('Percentage group 1: \n{}'.format(len(event1)/len(age_group1)))
print('Percentage group 2: \n{}'.format(len(event2)/len(age_group2)))
print('Percentage group 3: \n{}'.format(len(event3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Event'].values), age_group2['Event'].values, age_group3['Event'].values)))

# calculating gender
female1 = (age_group1[age_group1['Gender']==0])
female2 = (age_group2[age_group2['Gender']==0])
female3 = (age_group3[age_group3['Gender']==0])
print('Number of female in group 1:\n{}'.format(len(female1)))
print('Number of females in group 2:\n{}'.format(len(female2)))
print('Number of females in group 3:\n{}'.format(len(female3)))
print('Percentage female 1: \n{}'.format(len(female1)/len(age_group1)))
print('Percentage female 2: \n{}'.format(len(female2)/len(age_group2)))
print('Percentage female 3: \n{}'.format(len(female3)/len(age_group3)))

male1 = (age_group1[age_group1['Gender']==1])
male2 = (age_group2[age_group2['Gender']==1])
male3 = (age_group3[age_group3['Gender']==1])
print('Number of males in group 1:\n{}'.format(len(male1)))
print('Number of males in group 2:\n{}'.format(len(male2)))
print('Number of males in group 3:\n{}'.format(len(male3)))
print('Percentage male 1: \n{}'.format(len(male1)/len(age_group1)))
print('Percentage male 2: \n{}'.format(len(male2)/len(age_group2)))
print('Percentage male 3: \n{}'.format(len(male3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Gender'].values), age_group2['Gender'].values, age_group3['Gender'].values)))

# calculating smoking
smoke1 = (age_group1[age_group1['Smoking_history']==1])
smoke2 = (age_group2[age_group2['Smoking_history']==1])
smoke3 = (age_group3[age_group3['Smoking_history']==1])
print('Number of smoking in group 1:\n{}'.format(len(smoke1)))
print('Number of smoking in group 2:\n{}'.format(len(smoke2)))
print('Number of smoking in group 3:\n{}'.format(len(smoke3)))
print('Percentage smoking 1: \n{}'.format(len(smoke1)/len(age_group1)))
print('Percentage smoking 2: \n{}'.format(len(smoke2)/len(age_group2)))
print('Percentage smoking 3: \n{}'.format(len(smoke3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Smoking_history'].values), age_group2['Smoking_history'].values, age_group3['Smoking_history'].values)))

# calculating diabetes
dm1 = (age_group1[age_group1['Diabetes_mellitus_(disorder)']==1])
dm2 = (age_group2[age_group2['Diabetes_mellitus_(disorder)']==1])
dm3 = (age_group3[age_group3['Diabetes_mellitus_(disorder)']==1])
print('Number of DM in group 1:\n{}'.format(len(dm1)))
print('Number of DM in group 2:\n{}'.format(len(dm2)))
print('Number of DM in group 3:\n{}'.format(len(dm3)))
print('Percentage DM 1: \n{}'.format(len(dm1)/len(age_group1)))
print('Percentage DM 2: \n{}'.format(len(dm2)/len(age_group2)))
print('Percentage DM 3: \n{}'.format(len(dm3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Diabetes_mellitus_(disorder)'].values), age_group2['Diabetes_mellitus_(disorder)'].values, age_group3['Diabetes_mellitus_(disorder)'].values)))

# calculating hypertension
htn1 = (age_group1[age_group1['Essential_hypertension']==1])
htn2 = (age_group2[age_group2['Essential_hypertension']==1])
htn3 = (age_group3[age_group3['Essential_hypertension']==1])
print('Number of HTN in group 1:\n{}'.format(len(htn1)))
print('Number of HTN in group 2:\n{}'.format(len(htn2)))
print('Number of HTN in group 3:\n{}'.format(len(htn3)))
print('Percentage HTN 1: \n{}'.format(len(htn1)/len(age_group1)))
print('Percentage HTN 2: \n{}'.format(len(htn2)/len(age_group2)))
print('Percentage HTN 3: \n{}'.format(len(htn3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Essential_hypertension'].values), age_group2['Essential_hypertension'].values, age_group3['Essential_hypertension'].values)))

# calculating dyslipidaemia
lip1 = (age_group1[age_group1['Dyslipidaemia']==1])
lip2 = (age_group2[age_group2['Dyslipidaemia']==1])
lip3 = (age_group3[age_group3['Dyslipidaemia']==1])
print('Number of Dyslipidaemia in group 1:\n{}'.format(len(lip1)))
print('Number of Dyslipidaemia in group 2:\n{}'.format(len(lip2)))
print('Number of Dyslipidaemia in group 3:\n{}'.format(len(lip3)))
print('Percentage Dyslipidaemia 1: \n{}'.format(len(lip1)/len(age_group1)))
print('Percentage Dyslipidaemia 2: \n{}'.format(len(lip2)/len(age_group2)))
print('Percentage Dyslipidaemia 3: \n{}'.format(len(lip3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Dyslipidaemia'].values), age_group2['Dyslipidaemia'].values, age_group3['Dyslipidaemia'].values)))

# calculating CVA
cva1 = (age_group1[age_group1['CVA']==1])
cva2 = (age_group2[age_group2['CVA']==1])
cva3 = (age_group3[age_group3['CVA']==1])
print('Number of CVA in group 1:\n{}'.format(len(cva1)))
print('Number of CVA in group 2:\n{}'.format(len(cva2)))
print('Number of CVA in group 3:\n{}'.format(len(cva3)))
print('Percentage CVA 1: \n{}'.format(len(cva1)/len(age_group1)))
print('Percentage CVA 2: \n{}'.format(len(cva2)/len(age_group2)))
print('Percentage CVA 3: \n{}'.format(len(cva3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['CVA'].values), age_group2['CVA'].values, age_group3['CVA'].values)))

# calculating CKD
ckd1 = (age_group1[age_group1['Chronic_kidney_disease_(disorder)']==1])
ckd2 = (age_group2[age_group2['Chronic_kidney_disease_(disorder)']==1])
ckd3 = (age_group3[age_group3['Chronic_kidney_disease_(disorder)']==1])
print('Number of CKD in group 1:\n{}'.format(len(ckd1)))
print('Number of CKD in group 2:\n{}'.format(len(ckd2)))
print('Number of CKD in group 3:\n{}'.format(len(ckd3)))
print('Percentage CKD 1: \n{}'.format(len(ckd1)/len(age_group1)))
print('Percentage CKD 2: \n{}'.format(len(ckd2)/len(age_group2)))
print('Percentage CKD 3: \n{}'.format(len(ckd3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Chronic_kidney_disease_(disorder)'].values), age_group2['Chronic_kidney_disease_(disorder)'].values, age_group3['Chronic_kidney_disease_(disorder)'].values)))

# calculating heart failure
hf1 = (age_group1[age_group1['Heart_failure_(disorder)']==1])
hf2 = (age_group2[age_group2['Heart_failure_(disorder)']==1])
hf3 = (age_group3[age_group3['Heart_failure_(disorder)']==1])
print('Number of HF in group 1:\n{}'.format(len(hf1)))
print('Number of HF in group 2:\n{}'.format(len(hf2)))
print('Number of HF in group 3:\n{}'.format(len(hf3)))
print('Percentage HF 1: \n{}'.format(len(hf1)/len(age_group1)))
print('Percentage HF 2: \n{}'.format(len(hf2)/len(age_group2)))
print('Percentage HF 3: \n{}'.format(len(hf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Heart_failure_(disorder)'].values), age_group2['Heart_failure_(disorder)'].values, age_group3['Heart_failure_(disorder)'].values)))

# calculating previous myocardial infarction
mi1 = (age_group1[age_group1['Myocardial_infarction_(disorder)']==1])
mi2 = (age_group2[age_group2['Myocardial_infarction_(disorder)']==1])
mi3 = (age_group3[age_group3['Myocardial_infarction_(disorder)']==1])
print('Number of MI in group 1:\n{}'.format(len(mi1)))
print('Number of MI in group 2:\n{}'.format(len(mi2)))
print('Number of MI in group 3:\n{}'.format(len(mi3)))
print('Percentage MI 1: \n{}'.format(len(mi1)/len(age_group1)))
print('Percentage MI 2: \n{}'.format(len(mi2)/len(age_group2)))
print('Percentage MI 3: \n{}'.format(len(mi3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Myocardial_infarction_(disorder)'].values), age_group2['Myocardial_infarction_(disorder)'].values, age_group3['Myocardial_infarction_(disorder)'].values)))

# calculating atrial fibrillation
af1 = (age_group1[age_group1['Atrial_fibrillation_(disorder)']==1])
af2 = (age_group2[age_group2['Atrial_fibrillation_(disorder)']==1])
af3 = (age_group3[age_group3['Atrial_fibrillation_(disorder)']==1])
print('Number of AF in group 1:\n{}'.format(len(af1)))
print('Number of AF in group 2:\n{}'.format(len(af2)))
print('Number of AF in group 3:\n{}'.format(len(af3)))
print('Percentage AF 1: \n{}'.format(len(af1)/len(age_group1)))
print('Percentage AF 2: \n{}'.format(len(af2)/len(age_group2)))
print('Percentage AF 3: \n{}'.format(len(af3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Atrial_fibrillation_(disorder)'].values), age_group2['Atrial_fibrillation_(disorder)'].values, age_group3['Atrial_fibrillation_(disorder)'].values)))

# calculating atrial flutter
afl1 = (age_group1[age_group1['Atrial_flutter_(disorder)']==1])
afl2 = (age_group2[age_group2['Atrial_flutter_(disorder)']==1])
afl3 = (age_group3[age_group3['Atrial_flutter_(disorder)']==1])
print('Number of AFL in group 1:\n{}'.format(len(afl1)))
print('Number of AFL in group 2:\n{}'.format(len(afl2)))
print('Number of AFL in group 3:\n{}'.format(len(afl3)))
print('Percentage AFL 1: \n{}'.format(len(afl1)/len(age_group1)))
print('Percentage AFL 2: \n{}'.format(len(afl2)/len(age_group2)))
print('Percentage AFL 3: \n{}'.format(len(afl3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Atrial_flutter_(disorder)'].values), age_group2['Atrial_flutter_(disorder)'].values, age_group3['Atrial_flutter_(disorder)'].values)))

# calculating 1st degree heart block
fhb1 = (age_group1[age_group1['First_degree_atrioventricular_block_(disorder)']==1])
fhb2 = (age_group2[age_group2['First_degree_atrioventricular_block_(disorder)']==1])
fhb3 = (age_group3[age_group3['First_degree_atrioventricular_block_(disorder)']==1])
print('Number of 1st HB in group 1:\n{}'.format(len(fhb1)))
print('Number of 1st HB in group 2:\n{}'.format(len(fhb2)))
print('Number of 1st HB in group 3:\n{}'.format(len(fhb3)))
print('Percentage 1st HB 1: \n{}'.format(len(fhb1)/len(age_group1)))
print('Percentage 1st HB 2: \n{}'.format(len(fhb2)/len(age_group2)))
print('Percentage 1st HB 3: \n{}'.format(len(fhb3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['First_degree_atrioventricular_block_(disorder)'].values), age_group2['First_degree_atrioventricular_block_(disorder)'].values, age_group3['First_degree_atrioventricular_block_(disorder)'].values)))

# calculating 2nd degree heart block
shb1 = (age_group1[age_group1['Second_degree_atrioventricular_block_(disorder)']==1])
shb2 = (age_group2[age_group2['Second_degree_atrioventricular_block_(disorder)']==1])
shb3 = (age_group3[age_group3['Second_degree_atrioventricular_block_(disorder)']==1])
print('Number of 2nd HB in group 1:\n{}'.format(len(shb1)))
print('Number of 2nd HB in group 2:\n{}'.format(len(shb2)))
print('Number of 2nd HB in group 3:\n{}'.format(len(shb3)))
print('Percentage 2nd HB 1: \n{}'.format(len(shb1)/len(age_group1)))
print('Percentage 2nd HB 2: \n{}'.format(len(shb2)/len(age_group2)))
print('Percentage 2nd HB 3: \n{}'.format(len(shb3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Second_degree_atrioventricular_block_(disorder)'].values), age_group2['Second_degree_atrioventricular_block_(disorder)'].values, age_group3['Second_degree_atrioventricular_block_(disorder)'].values)))

# calculating 3rd degree heart block
chb1 = (age_group1[age_group1['Complete_atrioventricular_block_(disorder)']==1])
chb2 = (age_group2[age_group2['Complete_atrioventricular_block_(disorder)']==1])
chb3 = (age_group3[age_group3['Complete_atrioventricular_block_(disorder)']==1])
print('Number of 3rd HB in group 1:\n{}'.format(len(chb1)))
print('Number of 3rd HB in group 2:\n{}'.format(len(chb2)))
print('Number of 3rd HB in group 3:\n{}'.format(len(chb3)))
print('Percentage 3rd HB 1: \n{}'.format(len(chb1)/len(age_group1)))
print('Percentage 3rd HB 2: \n{}'.format(len(chb2)/len(age_group2)))
print('Percentage 3rd HB 3: \n{}'.format(len(chb3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Complete_atrioventricular_block_(disorder)'].values), age_group2['Complete_atrioventricular_block_(disorder)'].values, age_group3['Complete_atrioventricular_block_(disorder)'].values)))

# calculating VT
vt1 = (age_group1[age_group1['Ventricular_tachycardia_(disorder)']==1])
vt2 = (age_group2[age_group2['Ventricular_tachycardia_(disorder)']==1])
vt3 = (age_group3[age_group3['Ventricular_tachycardia_(disorder)']==1])
print('Number of VT in group 1:\n{}'.format(len(vt1)))
print('Number of VT in group 2:\n{}'.format(len(vt2)))
print('Number of VT in group 3:\n{}'.format(len(vt3)))
print('Percentage VT 1: \n{}'.format(len(vt1)/len(age_group1)))
print('Percentage VT 2: \n{}'.format(len(vt2)/len(age_group2)))
print('Percentage VT 3: \n{}'.format(len(vt3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Ventricular_tachycardia_(disorder)'].values), age_group2['Ventricular_tachycardia_(disorder)'].values, age_group3['Ventricular_tachycardia_(disorder)'].values)))

# calculating VF
vf1 = (age_group1[age_group1['Ventricular_fibrillation_(disorder)']==1])
vf2 = (age_group2[age_group2['Ventricular_fibrillation_(disorder)']==1])
vf3 = (age_group3[age_group3['Ventricular_fibrillation_(disorder)']==1])
print('Number of VF in group 1:\n{}'.format(len(vf1)))
print('Number of VF in group 2:\n{}'.format(len(vf2)))
print('Number of VF in group 3:\n{}'.format(len(vf3)))
print('Percentage VF 1: \n{}'.format(len(vf1)/len(age_group1)))
print('Percentage VF 2: \n{}'.format(len(vf2)/len(age_group2)))
print('Percentage VF 3: \n{}'.format(len(vf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Ventricular_fibrillation_(disorder)'].values), age_group2['Ventricular_fibrillation_(disorder)'].values, age_group3['Ventricular_fibrillation_(disorder)'].values)))

# calculating field strength
lowf1 = (age_group1[age_group1['Field_strength']==0])
lowf2 = (age_group2[age_group2['Field_strength']==0])
lowf3 = (age_group3[age_group3['Field_strength']==0])
print('Number of 1.5T in group 1:\n{}'.format(len(lowf1)))
print('Number of 1.5T in group 2:\n{}'.format(len(lowf2)))
print('Number of 1.5T in group 3:\n{}'.format(len(lowf3)))
print('Percentage 1.5T 1: \n{}'.format(len(lowf1)/len(age_group1)))
print('Percentage 1.5T 2: \n{}'.format(len(lowf2)/len(age_group2)))
print('Percentage 1.5T 3: \n{}'.format(len(lowf3)/len(age_group3)))
hf1 = (age_group1[age_group1['Field_strength']==1])
hf2 = (age_group2[age_group2['Field_strength']==1])
hf3 = (age_group3[age_group3['Field_strength']==1])
print('Number of 3T in group 1:\n{}'.format(len(hf1)))
print('Number of 3T in group 2:\n{}'.format(len(hf2)))
print('Number of 3T in group 3:\n{}'.format(len(hf3)))
print('Percentage 3T 1: \n{}'.format(len(hf1)/len(age_group1)))
print('Percentage 3T 2: \n{}'.format(len(hf2)/len(age_group2)))
print('Percentage 3T 3: \n{}'.format(len(hf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Field_strength'].values), age_group2['Field_strength'].values, age_group3['Field_strength'].values)))

# calculating stress agent
a1 = (age_group1[age_group1['Stress_agent']==0])
a2 = (age_group2[age_group2['Stress_agent']==0])
a3 = (age_group3[age_group3['Stress_agent']==0])
print('Number of Adenosine in group 1:\n{}'.format(len(a1)))
print('Number of Adenosine in group 2:\n{}'.format(len(a2)))
print('Number of Adenosine in group 3:\n{}'.format(len(a3)))
print('Percentage Adenosine1: \n{}'.format(len(a1)/len(age_group1)))
print('Percentage Adenosine 2: \n{}'.format(len(a2)/len(age_group2)))
print('Percentage Adenosine 3: \n{}'.format(len(a3)/len(age_group3)))
reg1 = (age_group1[age_group1['Stress_agent']==1])
reg2 = (age_group2[age_group2['Stress_agent']==1])
reg3 = (age_group3[age_group3['Stress_agent']==1])
print('Number of Regadenosine in group 1:\n{}'.format(len(reg1)))
print('Number of Regadenosine in group 2:\n{}'.format(len(reg2)))
print('Number of Regadenosine in group 3:\n{}'.format(len(reg3)))
print('Percentage Regadenosine 1: \n{}'.format(len(reg1)/len(age_group1)))
print('Percentage Regadenosine 2: \n{}'.format(len(reg2)/len(age_group2)))
print('Percentage Regadenosine 3: \n{}'.format(len(reg3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Stress_agent'].values), age_group2['Stress_agent'].values, age_group3['Stress_agent'].values)))

# calculating perfusion
perf1 = (age_group1[age_group1['Positive_perf']==1])
perf2 = (age_group2[age_group2['Positive_perf']==1])
perf3 = (age_group3[age_group3['Positive_perf']==1])
print('Number of +ve perf in group 1:\n{}'.format(len(perf1)))
print('Number of +ve perf in group 2:\n{}'.format(len(perf2)))
print('Number of +ve perf in group 3:\n{}'.format(len(perf3)))
print('Percentage +ve perf 1: \n{}'.format(len(perf1)/len(age_group1)))
print('Percentage +ve perf 2: \n{}'.format(len(perf2)/len(age_group2)))
print('Percentage +ve perf 3: \n{}'.format(len(perf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Positive_perf'].values), age_group2['Positive_perf'].values, age_group3['Positive_perf'].values)))

# calculating LGE
lge1 = (age_group1[age_group1['Positive_LGE']==1])
lge2 = (age_group2[age_group2['Positive_LGE']==1])
lge3 = (age_group3[age_group3['Positive_LGE']==1])
print('Number of +ve LGE in group 1:\n{}'.format(len(lge1)))
print('Number of +ve LGE in group 2:\n{}'.format(len(lge2)))
print('Number of +ve LGE in group 3:\n{}'.format(len(lge3)))
print('Percentage +ve LGE 1: \n{}'.format(len(lge1)/len(age_group1)))
print('Percentage +ve LGE 2: \n{}'.format(len(lge2)/len(age_group2)))
print('Percentage +ve LGE 3: \n{}'.format(len(lge3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Positive_LGE'].values), age_group2['Positive_LGE'].values, age_group3['Positive_LGE'].values)))

# calculating LV function
lvm1 = pd.Series(data=age_group1['LVEF_(%)'].astype('int'))
print('LVEF mean for group 1:\n{}'.format(lvm1.mean()))
print('LVEF SD for group 1:\n{}'.format(lvm1.std()))
lvm2 = pd.Series(data=age_group2['LVEF_(%)'].astype('int'))
print('LVEF mean for group 2:\n{}'.format(lvm2.mean()))
print('LVEF SD for group 2:\n{}'.format(lvm2.std()))
lvm3 = pd.Series(data=age_group3['LVEF_(%)'].astype('int'))
print('LVEF mean for group 3:\n{}'.format(lvm3.mean()))
print('LVEF SD for group 3:\n{}'.format(lvm3.std()))
print('One Way ANOVA test LVEF: \n{}'.format(stats.f_oneway(lvm1, lvm2, lvm3)))
lvm = pd.Series(data=data['LVEF_(%)'].astype('int'))
print('LVEF mean for total:\n{}'.format(lvm.mean()))
print('LVEF SD for total:\n{}'.format(lvm.std()))

# calculating RV function
rvm1 = pd.Series(data=age_group1['RVEF_(%)'].astype('int'))
print('RVEF mean for group 1:\n{}'.format(rvm1.mean()))
print('RVEF SD for group 1:\n{}'.format(rvm1.std()))
rvm2 = pd.Series(data=age_group2['RVEF_(%)'].astype('int'))
print('RVEF mean for group 2:\n{}'.format(rvm2.mean()))
print('RVEF SD for group 2:\n{}'.format(rvm2.std()))
rvm3 = pd.Series(data=age_group3['RVEF_(%)'].astype('int'))
print('RVEF mean for group 3:\n{}'.format(rvm3.mean()))
print('RVEF SD for group 3:\n{}'.format(rvm3.std()))
print('One Way ANOVA test RVEF: \n{}'.format(stats.f_oneway(rvm1, rvm2, rvm3)))
rvm = pd.Series(data=data['RVEF_(%)'].astype('int'))
print('RVEF mean for total:\n{}'.format(rvm.mean()))
print('RVEF SD for total:\n{}'.format(rvm.std()))

# plotting arrhythmia
x = ('Asystole', 'Atrial fibrillation', 'Atrial flutter', '1st degree AV block', '2nd degree AV block', '3rd degree AV block', 'Sick sinus syndrome', 'Ventricular tachycardia', 'Ventricular fibrillation')
y = [6, 485, 135, 30, 118, 4, 2, 39, 244]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y)
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

# plotting clinical risk factors
x = ('Smoking history','Cerebrovascular accident', 'Chronic kidney disease', 'Diabetes mellitus', 'Dyslipidemia', 'Heart failure', 'Myocardial infarction', 'Transient ischemic attack', 'Essential hypertension')
y = [487, 257, 170, 136, 691, 517, 800, 150, 1264]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='purple')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

# plotting perfusion
print(data['LAD_perf'].value_counts())
print(data['LCx_perf'].value_counts())
print(data['RCA_perf'].value_counts())
x = ('LAD ischaemia', 'LCx ischaemia', 'RCA ischaemia')
y = [621, 672, 712]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='slateblue')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

# plotting LGE
print(data['LAD_LGE'].value_counts())
print(data['LCx_LGE'].value_counts())
print(data['RCA_LGE'].value_counts())
x = ('LAD LGE', 'LCx LGE', 'RCA LGE')
y = [686, 829, 921]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='yellowgreen')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()


# scatter plot
data['Patient Gender'] = data['patient_GenderCode']
plt.rcParams.update({'font.size': 16})
sns.set(font_scale=1.3)
fig, axs = plt.subplots(1,2)
bins = [0,65,75,np.inf]
names = ['<65', '65-75', '>75']
data['Age'] = pd.cut(data['Age_on_20.08.2021'], bins, labels=names)
sns.catplot(x="Age", y="Positive_perf", hue='Patient Gender', kind="bar", data=data, ax=axs[0])
plt.text(1, 0.45, "p value <0.001", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.ylabel('Positive stress perfusion', fontsize=16)
plt.xlabel('Age', fontsize=16)
sns.catplot(x="Age", y="Positive_LGE", hue='Patient Gender', kind="bar", data=data, ax=axs[1])
plt.text(1, 0.8, "p value <0.001", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.ylabel('Positive ischaemic LGE', fontsize=16)
plt.xlabel('Age', fontsize=16)
plt.show()

