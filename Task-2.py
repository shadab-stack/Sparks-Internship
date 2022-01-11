#!/usr/bin/env python
# coding: utf-8

# # The Sparks foundation Task 2
# 
# 
# ## Data science & Business Analytics tasks
# 
# 
# ### Prediction using Unsupervised ML
# 
# #### Shadab Shiekh

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()


# In[3]:


iris = datasets.load_iris()

iris.feature_names


# In[4]:


iris


# In[5]:



iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

# See the first 5 rows
iris_df.head(10)


# In[6]:


X = iris.data[:, :2]
y = iris.target
X.shape, y.shape


# In[7]:


plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Sepa1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18);


# In[8]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values
x


# In[10]:


from sklearn.cluster import KMeans
wcss = []


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[11]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[12]:


KMeans


# In[13]:


km = KMeans(n_clusters = 3, random_state=21)
km.fit(X)


# In[14]:


centers = km.cluster_centers_
print(centers)


# In[15]:


new_labels = km.labels_


# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18);


# In[ ]:




