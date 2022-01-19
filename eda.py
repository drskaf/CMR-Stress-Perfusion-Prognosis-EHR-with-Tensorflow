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
y = [10, 1070, 304, 244, 15, 72, 8, 127, 560]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y)
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

print(data['Cerebrovascular accident (disorder)'].value_counts())
print(data['Chronic kidney disease (disorder)'].value_counts())
print(data['Diabetes mellitus (disorder)'].value_counts())
print(data['Dyslipidaemia'].value_counts())
print(data['Heart failure (disorder)'].value_counts())
print(data['Myocardial infarction (disorder)'].value_counts())
print(data['Transient ischemic attack (disorder)'].value_counts())
print(data['Essential hypertension'].value_counts())
x = ('Cerebrovascular accident', 'Chronic kidney disease', 'Diabetes mellitus', 'Dyslipidemia', 'Heart failure', 'Myocardial infarction', 'Transient ischemic attack', 'Essential hypertension')
y = [503, 336, 259, 1106, 1127, 1461, 262, 2307]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='purple')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

mini_data = pd.read_csv('survivalm.csv')
print(mini_data['Asystole (disorder)'].value_counts())
print(mini_data['Atrial fibrillation (disorder)'].value_counts())
print(mini_data['Atrial flutter (disorder)'].value_counts())
print(mini_data['Complete atrioventricular block (disorder)'].value_counts())
print(mini_data['First degree atrioventricular block (disorder)'].value_counts())
print(mini_data['Second degree atrioventricular block (disorder)'].value_counts())
print(mini_data['Sick sinus syndrome (disorder)'].value_counts())
print(mini_data['Ventricular fibrillation (disorder)'].value_counts())
print(mini_data['Ventricular tachycardia (disorder)'].value_counts())
x = ('Asystole', 'Atrial fibrillation', 'Atrial flutter', '1st degree AV block', '2nd degree AV block', '3rd degree AV block', 'Sick sinus syndrome', 'Ventricular tachycardia', 'Ventricular fibrillation')
y = [0, 65, 20, 17, 0, 7, 0, 5, 24]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='orange')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

print(mini_data['Cerebrovascular accident (disorder)'].value_counts())
print(mini_data['Chronic kidney disease (disorder)'].value_counts())
print(mini_data['Diabetes mellitus (disorder)'].value_counts())
print(mini_data['Dyslipidaemia'].value_counts())
print(mini_data['Heart failure (disorder)'].value_counts())
print(mini_data['Myocardial infarction (disorder)'].value_counts())
print(mini_data['Transient ischemic attack (disorder)'].value_counts())
print(mini_data['Essential hypertension'].value_counts())
x = ('Cerebrovascular accident', 'Chronic kidney disease', 'Diabetes mellitus', 'Dyslipidemia', 'Heart failure', 'Myocardial infarction', 'Transient ischemic attack', 'Essential hypertension')
y = [31, 46, 18, 70, 83, 69, 18, 118]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='skyblue')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

print(mini_data['LAD_perf'].value_counts())
print(mini_data['LCx_perf'].value_counts())
print(mini_data['RCA_perf'].value_counts())
print(mini_data['MVD'].value_counts())
x = ('LAD ischaemia', 'LCx ischaemia', 'RCA ischaemia', 'Microvascular ischaemia')
y = [59, 71, 75, 23]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='slateblue')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()

print(mini_data['LAD_LGE'].value_counts())
print(mini_data['LCx_LGE'].value_counts())
print(mini_data['RCA_LGE'].value_counts())
x = ('LAD LGE', 'LCx LGE', 'RCA LGE')
y = [81, 90, 108]
y_pos = np.arange(len(x))
bars = plt.barh(y_pos, y, color='yellowgreen')
for  bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y, s=f'{width}')
plt.yticks(y_pos, x)
plt.show()
