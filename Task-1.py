#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation Task 1
# ###                         Shadab Shiekh
# ## Linear Regression with Python Scikit Learn
# #####  linear regression involving two variables.
# ###### Predicting the percentage of an student based on the no. of study hours. This is a simple linear regression as it involves just 2 variables.
# 

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(10)


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.plot(x='Hours', y='Scores', style='x')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[7]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[19]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[20]:



y_pred = regressor.predict(X_test)


# In[21]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[22]:


# Required result
hours = 9.25
pred = regressor.predict([[9.25]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# In[23]:


from sklearn import metrics  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




