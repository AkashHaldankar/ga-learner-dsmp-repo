# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
print(df.columns)
list_of_columns_to_exclude = ['customerID','Churn']
X = df.drop(list_of_columns_to_exclude, axis=1)
print(X.shape)
y = df['Churn'].copy()

#Splitting into train and test dataset
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

def convert_float(val):
    """
    Convert the string number value to a float
     - Remove $
     - Remove commas
     - Convert to float type
    """
    # new_val = val.replace(',','').replace('$', '')
    return float(val)

# Code starts here
X_train['TotalCharges'].replace(' ', np.NaN, inplace=True)
X_test['TotalCharges'].replace(' ', np.NaN, inplace=True)

print(X_train.dtypes)
s = X_train['TotalCharges'].copy()
s = pd.to_numeric(s)
print("S type {0}".format(s.dtypes))
#X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'].str.replace(' ',''), errors='force')
#X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'].str.replace(' ',''), errors='force')

#print(X_train.shape)
#print(X_train['TotalCharges'].describe())
#print(X_train['TotalCharges'].isnull().sum())

# Converting the column to `float` datatype
X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'])
#X_train['TotalCharges'] = X_train['TotalCharges'].astype(dtype=float)
X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'])
# X_test['TotalCharges'] = X_test['TotalCharges'].astype(dtype=float)
#X_train['TotalCharges'].apply(convert_float)
# X_test['TotalCharges'].apply(convert_float)

# Fill NA with mean
X_train['TotalCharges'].fillna(value= X_train['TotalCharges'].mean(),inplace= True)
X_test['TotalCharges'].fillna(value= X_test['TotalCharges'].mean(),inplace= True)

#Sum of null values of each column
total_null_X_train = X_train.isnull().sum()
count_null_X_train = X_train.isnull().count()

# Label Encoder
label_encoder = LabelEncoder()
cat_cols = X_train.select_dtypes(include=["category","object_"]).columns
print(cat_cols)
for each_cat_col in cat_cols:
    X_train[each_cat_col] = label_encoder.fit_transform(X_train[each_cat_col])
    X_test[each_cat_col] = label_encoder.fit_transform(X_test[each_cat_col])
#X_train['TotalCharges'] = label_encoder.fit_transform(X_train['TotalCharges'])
#X_test['TotalCharges'] = label_encoder.fit_transform(X_test['TotalCharges'])

dict_replace = {'No':0, 'Yes':1}
y_train.replace({'No':0, 'Yes':1},inplace=True)
y_test.replace({'No':0, 'Yes':1},inplace=True)

print(X_train.dtypes)
print(X_train.head())


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train, X_test, y_train, y_test)

# Fitting of weak classifier with Adaboost
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test,y_pred)
print(ada_score)

ada_cm = confusion_matrix(y_test,y_pred)
print(ada_cm)

ada_cr = classification_report(y_test,y_pred)
print(ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

# Fitting of weak classifier with XGBoost
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test,y_pred)
print(xgb_score)

xgb_cm = confusion_matrix(y_test,y_pred)
print(xgb_cm)

xgb_cr = classification_report(y_test,y_pred)
print(xgb_cr)

# Grid Search
clf_model = GridSearchCV(estimator=xgb_model,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test,y_pred)
print(clf_score)

clf_cm = confusion_matrix(y_test,y_pred)
print(clf_cm)

clf_cr = classification_report(y_test,y_pred)
print(clf_cr)


