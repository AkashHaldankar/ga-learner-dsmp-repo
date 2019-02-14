# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)

print(df.head())

print(df.info())


clean_str_cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for each_col in clean_str_cols:
    df[each_col] = df[each_col].str.replace('$','')
    df[each_col] = df[each_col].str.replace(',','')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

count = y.value_counts()

print(count)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6)


# Code ends here


# --------------
# Code starts here
float_cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for each_col in float_cols:
    X_train[each_col] = X_train[each_col].astype(float)
    X_test[each_col] = X_test[each_col].astype(float)

print(X_train.head())


missing_columns = (X_train.isnull().sum()*100) / len(X_train)
mask = missing_columns > 0
columns = missing_columns[mask].index.tolist()
print(columns)
# Code ends here


# --------------
# Code starts here
drop_miss_cols = ['YOJ','OCCUPATION']


X_train.dropna(subset=drop_miss_cols,inplace=True)
X_test.dropna(subset=drop_miss_cols,inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

mean_miss_cols = ['AGE','CAR_AGE','INCOME','HOME_VAL']

for each_mean_miss_col in mean_miss_cols:
    X_train[mean_miss_cols].fillna(X_train[mean_miss_cols].mean(),inplace=True)
    X_test[mean_miss_cols].fillna(X_test[mean_miss_cols].mean(),inplace=True)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()

for each_col in columns:
    X_train[each_col] = le.fit_transform(X_train[each_col])
    X_test[each_col] = le.fit_transform(X_test[each_col])

print(X_train.head())
# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# code starts here 
model = LogisticRegression(random_state=0)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

print(score,precision)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=6)
X_train,y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)

print(score)
# Code ends here


