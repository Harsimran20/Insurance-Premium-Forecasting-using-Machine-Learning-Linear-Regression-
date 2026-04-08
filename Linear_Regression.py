#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import seaborn as sns


# In[7]:


insurance_data = pd.read_csv(r"C:\Users\ACER\Downloads\archive (9)\insurance.csv")


# In[8]:


insurance_data


# In[9]:


sns.scatterplot(x=insurance_data["bmi"],y=insurance_data["charges"],hue = insurance_data["smoker"])


# In[10]:


X = insurance_data.drop(columns = ["charges","region"])
y = insurance_data["charges"]
X["sex"] = X["sex"].map({"female":1, "male":0})
X["smoker"] = X["smoker"].map({"yes":1, "no":0})


# In[11]:


X.head()


# In[12]:


y.head()


# In[13]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# In[22]:


X_test.head()


# In[24]:


# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[25]:


# Predict Values
y_pred = model.predict(X_test)


# In[28]:


y_pred


# In[30]:


y_test


# In[36]:


# Evaluate 
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("r-squared:", r2)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print("Adjusted r^2:", adjusted_r2)


# In[38]:


X_test.shape


# In[ ]:




