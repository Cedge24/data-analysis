#!/usr/bin/env python
# coding: utf-8

# Conor Edgecumbe
# 10.27.21
# Stout Case Study #1 : Lending Club

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[43]:


df = pd.read_csv('DATA\loans_full_schema.csv')


# # Cursory Analysis

# In[3]:


print(df.shape)
print(df.columns)
df.head()


# In[4]:


df.info()
df.describe()


# In[5]:


# checking missing values
df.isna().sum()


# # Feature Encoding and Selection

# In[7]:


# dropping features
df = df.drop('sub_grade',1)
df = df.drop('emp_title',1)
df = df.drop('state',1)
# dropping sparsely populated features
df.drop(['annual_income_joint','verification_income_joint','debt_to_income_joint'],axis=1,inplace=True)


# In[8]:


# printing distinct values of categorical features
dt_series = df.dtypes.sort_values(ascending=False)
print(dt_series.value_counts())
print('=========================')
for ind, val in dt_series.iteritems():
    if val == 'object':
        print(df[ind].value_counts())
        print('-------------------------')    


# In[9]:


# issue_month
enc = OrdinalEncoder(categories=[['Jan-2018','Feb-2018','Mar-2018']])
df[['issue_month']] = enc.fit_transform(df[['issue_month']])

# loan_status
enc = OrdinalEncoder(categories=[['Current','Fully Paid','In Grace Period','Late (31-120 days)','Late (16-30 days)','Late (16-30 days)','Charged Off']])
df[['loan_status']] = enc.fit_transform(df[['loan_status']])

# disbursement_method
enc = OrdinalEncoder()
df[['disbursement_method']] = enc.fit_transform(df[['disbursement_method']])

# grade
enc = OrdinalEncoder(categories=[['G','F','E','D','C','B','A']])
df[['grade']] = enc.fit_transform(df[['grade']])

# initial_listing_status
enc = OrdinalEncoder()
df[['initial_listing_status']] = enc.fit_transform(df[['initial_listing_status']])

# homeownership
enc = OrdinalEncoder(categories=[['RENT','MORTGAGE','OWN']])
df[['homeownership']] = enc.fit_transform(df[['homeownership']])

# verified_income PANDAS ONEHOT
onehot = pd.get_dummies(df['verified_income'])
df = df.drop('verified_income',axis=1)
df = df.join(onehot)
df = df.rename(columns={'Source Verified': 'source_verified_income','Not Verified':'not_verified_income','Verified':'verified_income'})

# application_type PANDAS ONEHOT
onehot = pd.get_dummies(df['application_type'])
df = df.drop('application_type',axis=1)
df = df.join(onehot)
df = df.rename(columns={'individual': 'individual_app','joint':'joint_app'})

# loan_purpose PANDAS ONEHOT
onehot = pd.get_dummies(df['loan_purpose'])
df = df.drop('loan_purpose',axis=1)
df = df.join(onehot)


# # Data Cleaning

# In[10]:


# handling missing values, I believe 0 adheres to domain of each remaining feature with missing values
df.fillna(value=0,inplace=True)


# In[11]:


# consistent dtypes
for c in df.columns:
    if df[c].dtype != 'float64':
        df[c] = df[c].astype(np.float64)


# # Data Scaling / Splitting

# In[12]:


# Standardization
scaler = StandardScaler()
df_standard = df.drop('interest_rate',1)
cols = df_standard.columns.tolist()
df_standard = scaler.fit_transform(df_standard)
df_standard = pd.DataFrame(df_standard, columns=cols)


# In[13]:


# extracting targets
y = df['interest_rate']
X = df_standard

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# Feature Analysis

# In[14]:


# Feature Selection with KBest Sklearn and Mutual Info Regression
selector = SelectKBest(score_func=mutual_info_regression,k=15)
selector.fit(X_train,y_train)
X_train_k = selector.transform(X_train)
X_test_k = selector.transform(X_test)


# In[40]:


# plot features
print(len(selector.scores_))
colors = {'installment':'r','grade':'c','paid_principal':'m'}
labels = list(colors.keys())
handles = [plt.patches.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.pyplot.legend(handles, labels, title='Most Informative Features', loc='upper right')
barlist = plt.pyplot.bar([i for i in range(len(selector.scores_))], selector.scores_)
plt.pyplot.figure(figsize=(15, 12), dpi=80)
barlist[34].set_color('r')
barlist[35].set_color('c')
barlist[42].set_color('m')
plt.pyplot.show()


# In[16]:


# printing scores and column names
i=0
scores = []
for s in selector.scores_:
    scores.append([s,X_train.columns[i]])
    i += 1
    
scores.sort()
for x in scores:
    print(x)


# # Visualization

# In[56]:


# visualizing target
print(y.mean)

grid = sns.JointGrid(X_train.index, y_train, space=0, ratio=50, height=10)
grid.plot_joint(plt.pyplot.scatter, color="b")
plt.pyplot.axhline(y=14.07, linewidth=6, color='r')


# In[19]:


# Visualizing variable correlations with scatter
sns.set(rc={'figure.figsize':(11.7,8.27)})
p = sns.scatterplot(data=df,x='paid_principal',y='interest_rate').set_title('Grade vs. IR')


# In[45]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
p = sns.scatterplot(data=df,x='paid_principal',y='interest_rate').set_title('paid_principal vs. IR')


# In[46]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
p = sns.scatterplot(data=df,x='emp_length',y='interest_rate').set_title('emp_length vs. IR')


# # Predicting Interest Rate with Learning Models

# In[20]:


# LinearRegression Baseline Model
model = LinearRegression()
model.fit(X_train_k,y_train)
y_hat = model.predict(X_test_k)

# evaluation metrics
mse = mean_squared_error(y_test,y_hat)
r2 = r2_score(y_test,y_hat)
print('mse: %.05f' % mse)
print('r2: %.05f' % r2)


# In[21]:


# Decision Tree Regression
model = DecisionTreeRegressor()
model.fit(X_train_k, y_train)
y_hat = model.predict(X_test_k)

# evaluation metrics
mse = mean_squared_error(y_test,y_hat)
r2 = r2_score(y_test,y_hat)
print('mse: %.05f' % mse)
print('r2: %.05f' % r2)

# DT has lower MSE and higher R2 score, indicating better performance than LR. I believe that this is because of the outliers in the dataset, which disproportionally affect LR models


# In[22]:


# Support Vector Machine
model = SVR()
model.fit(X_train_k,y_train)
y_hat = model.predict(X_test_k)

# evaluation metrics
mse = mean_squared_error(y_test,y_hat)
r2 = r2_score(y_test,y_hat)
print('mse: %.05f' % mse)
print('r2: %.05f' % r2)

# Performed the worst, could be because SVM performs poorly on large datasets with a lot of noise


# # Model Comparison Visual

# In[39]:


df = pd.DataFrame({
    'Model': ['LR','DT','SVM'],
    'MSE': [1.237,1.024,1.384],
    'R2': [0.950,0.958,0.944]
})
fig, ax1 = plt.pyplot.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Model').rename(columns=str.title)
sns.barplot(x='Model', y='Value', hue='Variable', data=tidy, ax=ax1)
sns.despine(fig)

