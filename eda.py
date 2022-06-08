import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="ticks", color_codes=True)

data = pd.read_csv('final.csv')

sns.catplot(x="patient_GenderCode", kind="count", palette="ch:.25", data=data)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.hist(data['Age_on_20.08.2021'])
ax.set_xlabel('Age in years')
ax.set_ylabel('Count')
plt.show()

print(data['Asystole (disorder)'].value_counts())
print(data['Atrial fibrillation (disorder)'].value_counts())
print(data['Atrial flutter (disorder)'].value_counts())
print(data['Complete atrioventricular block (disorder)'].value_counts())
print(data['First degree atrioventricular block (disorder)'].value_counts())
print(data['Second degree atrioventricular block (disorder)'].value_counts())
print(data['Sick sinus syndrome (disorder)'].value_counts())
print(data['Ventricular fibrillation (disorder)'].value_counts())
print(data['Ventricular tachycardia (disorder)'].value_counts())
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

print(data['Cerebrovascular_accident_(disorder)'].value_counts())
print(data['Chronic_kidney_disease_(disorder)'].value_counts())
print(data['Diabetes_mellitus_(disorder)'].value_counts())
print(data['Dyslipidaemia'].value_counts())
print(data['Heart_failure_(disorder)'].value_counts())
print(data['Myocardial_infarction_(disorder)'].value_counts())
print(data['Transient_ischemic_attack_(disorder)'].value_counts())
print(data['Essential_hypertension'].value_counts())
x = ('Cerebrovascular accident', 'Chronic kidney disease', 'Diabetes mellitus', 'Dyslipidemia', 'Heart failure', 'Myocardial infarction', 'Transient ischemic attack', 'Essential hypertension')
y = [257, 170, 136, 691, 517, 800, 150, 1264]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='purple')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

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


