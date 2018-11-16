# --------------
# Code starts here

avg_loan_amount = banks.pivot_table(index=['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc='mean')
print(avg_loan_amount)

# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x:x/12)
banks['loan_term'] = banks['Loan_Amount_Term'].apply(lambda x:x/12)

big_loan_term_condition = banks['loan_term'] >= 25
big_loan_term_data = banks[big_loan_term_condition]
big_loan_term = big_loan_term_data.shape[0]
print(big_loan_term)
# code ends here


# --------------
# code ends here

columns_to_show = ['ApplicantIncome', 'Credit_History']
 
loan_groupby=banks.groupby(['Loan_Status'])[columns_to_show]

# Check the mean value 
mean_values=loan_groupby.agg([np.mean])

print(mean_values)

# code ends here


# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)

#Categorical Values
categorical_var = bank.select_dtypes(include='object')
print(categorical_var)

#Numerical Values
numerical_var = bank.select_dtypes(include='number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'],axis=1)

# Print is null count
print(banks.isnull().sum())

# Find mode of columns
bank_mode = banks.mode()
print(bank_mode)

# Fill NA with mode
banks.fillna(banks.mode().iloc[0],inplace=True)

#Check if is null
print(banks.isnull().values.any())

#code ends here


# --------------
# code starts here
self_employed_yes = banks['Self_Employed'] == 'Yes'
loan_status = banks['Loan_Status'] == 'Y'  
loan_approved_se_data = banks[self_employed_yes & loan_status]
loan_approved_se = loan_approved_se_data.shape[0]

self_employed_no = banks['Self_Employed'] == 'No'
loan_approved_nse_data = banks[self_employed_no & loan_status]
loan_approved_nse = loan_approved_nse_data.shape[0] 

Loan_Status = 614
percentage_se = (loan_approved_se/Loan_Status)*100
percentage_nse = (loan_approved_nse/Loan_Status)*100
print(loan_approved_se,loan_approved_nse,percentage_se,percentage_nse)

# code ends here


