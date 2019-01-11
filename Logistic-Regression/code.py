# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
print(df.head())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 6)
print(X_train,X_test,y_train,y_test)
# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
plt.figure(figsize=(14,8))
X_train.boxplot(column = 'bmi')
q_value = X_train['bmi'].quantile(0.95)
y_train_value_counts = y_train.value_counts()
print(y_train_value_counts)
# Code ends here


# --------------
# Code starts here

# Seaborn visualization library
import seaborn as sns

relation = X_train.corr()
print(relation)

# Create the default pairplot
sns.pairplot(relation)

# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols = ['children','sex','region','smoker']
fig, axes = plt.subplots(nrows=2,ncols=2)

for i in range(0,2):
    for j in range(0,2):
        col = cols[i*2+j]
        sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=parameters)
grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score = roc_auc_score(y_test,y_pred)
y_pred_proba = grid.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test,y_pred_proba)
#plt.plot(fpr,tpr,label="Logistic model", auc=roc_auc)
# Code ends here


