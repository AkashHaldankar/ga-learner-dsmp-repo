# --------------
#Code starts here
fig, (ax_1,ax_2,ax_3) = plt.subplots(nrows=3,ncols=1)

ax_1.boxplot(data['Intelligence'])
ax_1.set_title('Intelligence')

ax_2.boxplot(data['Speed'])
ax_2.set_title('Speed')

ax_3.boxplot(data['Power'])
ax_3.set_title('Power')


# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
alignment.plot(kind='pie')


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best = data[data['Total'] > total_high]
super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
sc_df = data[['Strength','Combat']]
sc_covariance = data['Strength'].cov(data['Combat'])
sc_strength = data['Strength'].std()
sc_combat = data['Combat'].std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)

ic_df = data[['Intelligence','Combat']]
ic_covariance = data['Intelligence'].cov(data['Combat'])
ic_intelligence = data['Intelligence'].std()
ic_combat = data['Combat'].std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)

print(sc_pearson,ic_pearson)


