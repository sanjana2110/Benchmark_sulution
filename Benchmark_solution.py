#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np                   
import seaborn as sns              
import matplotlib.pyplot as plt 
import seaborn as sn                  
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.columns


# In[4]:


test.columns


# In[5]:


train.shape, test.shape


# In[6]:


train.dtypes


# In[7]:


train.head()


# # Univariate Analysis

# In[8]:


train['subscribed'].value_counts()


# In[9]:


train['subscribed'].value_counts(normalize=True)


# In[10]:


train['subscribed'].value_counts().plot.bar()


# In[11]:


sn.distplot(train["age"])


# In[12]:


train['job'].value_counts().plot.bar()


# In[13]:


train['default'].value_counts().plot.bar()


# # Bivariate Analysis

# In[14]:


print(pd.crosstab(train['job'],train['subscribed']))

job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# In[15]:


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


# In[16]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[17]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[18]:


train.isnull().sum()


# # Model Building

# In[19]:


target = train['subscribed']
train = train.drop('subscribed',1)


# In[20]:


train = pd.get_dummies(train)


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


# # Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression


# In[24]:


lreg = LogisticRegression()


# In[25]:


lreg.fit(X_train,y_train)


# In[26]:


prediction = lreg.predict(X_val)


# In[27]:


from sklearn.metrics import accuracy_score


# In[28]:


accuracy_score(y_val, prediction)


# # Decision Tree

# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[31]:


clf.fit(X_train,y_train)


# In[32]:


predict = clf.predict(X_val)


# In[33]:


accuracy_score(y_val, predict)


# In[34]:


test = pd.get_dummies(test)


# In[35]:


test_prediction = clf.predict(test)

