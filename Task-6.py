#!/usr/bin/env python
# coding: utf-8

# # # The Spark Foundation task - 6
# 
# #### Prediction using Decision Tree Algorithm
# 
# #### The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# #### Dataset link :- https://bit.ly/3kXTdox
# 
# ##### Author - Shadab Shiekh

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as datasets #for import dataset from sklearn library
import warnings
warnings.filterwarnings(action='ignore') #for ignore warnings


# In[36]:


#importing our iris datset
iris=datasets.load_iris() #loading the iris dataset

df= pd.DataFrame(iris.data, columns=iris.feature_names) # forming the iris dataset

df.head()


# In[37]:


new = iris.target
new


# In[38]:


df['species']=iris['target']
df['species']=df['species'].apply(lambda x: iris['target_names'][x])
df.head()


# In[39]:


df.shape


# In[40]:


df.describe()


# In[41]:


df.isnull().sum()


# In[42]:


sns.pairplot(df)


# In[43]:


# now turn to compare with various features and relationship between columns

sns.pairplot(df, hue='species')


# # 1. iris - setosa can easily be identified whereas the iris - versicolor & iris - virginics are overlapping.
# 

# In[44]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True) 


# # 2. Petal-length and Petal Width is important feature for identifying the flower

# In[45]:


# Ml is can not work with string values so we need to drop some columns. so we need to label encoding with 'y'

#x-y split 
x= df.iloc[:,:-1].values #feature matrix
y=df.iloc[:,-1].values #vector of predictions


# In[46]:


from sklearn.preprocessing import LabelEncoder # import library for label encoding


# In[47]:


y


# In[48]:


lab = LabelEncoder()
y= lab.fit_transform(y)
y


# In[49]:


from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(x, y, train_size=0.8, random_state=1)

x_train.shape, x_test.shape, y_train.shape, y_test.shape 


# In[50]:


#training the model

from sklearn.tree import DecisionTreeClassifier as dt

classifier= dt(class_weight='balanced')
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)


# In[51]:


y_pred


# In[52]:



data = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
data.head()


# In[53]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm= confusion_matrix(y_test,y_pred)
cm


# In[54]:


accuracy_score(y_test,y_pred)


# In[55]:


#Install required libraries
#pip install pydotplus
#pip install graphviz


# In[56]:


fn=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']


# In[57]:


# Visualize the graph or decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20,15))
plot_tree(classifier, feature_names=fn, class_names=cn, filled=True, rounded=True)
plt.show()


# In[ ]:





# In[ ]:




