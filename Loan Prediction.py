import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("trainloan.csv")
test = pd.read_csv("testloan.csv")
train_original = train.copy()
test_original = test.copy()

print(train.columns, test.columns)

print(train.dtypes)

print(train.shape, test.shape)

print(train['Loan_Status'].value_counts())
print(train['Loan_Status'].value_counts(normalize=True))

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')

plt.subplot(223)
train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')

plt.subplot(224)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')

plt.show()

print(train.isnull().sum())

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

print(train['Loan_Amount_Term'].value_counts())
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

table = train.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
table = test.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

x = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

x = pd.get_dummies(x)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.3, random_state=123)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LR = LogisticRegression()
LR.fit(x_train, y_train)

pred_cv = LR.predict(x_cv)
print(accuracy_score(y_cv, pred_cv))
