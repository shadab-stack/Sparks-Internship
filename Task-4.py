#!/usr/bin/env python
# coding: utf-8

#  # The Spark Foundation Task - 4
# 
# ### Perform ‘Exploratory Data Analysis’ on dataset ‘Global Terrorism’
# ### As a security/defense analyst, try to find out the hot zone of terrorism.
# ### What all security issues and insights you can derive by EDA?
# 
# #### Dataset link - https://bit.ly/2TK5Xn5
# 
# ##### Author - Shadab Shiekh

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings as wg


# In[2]:


df=pd.read_csv('globalterrorismdb_0718dist.csv',encoding='latin1',low_memory=False)
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


df.columns.values


# In[7]:


df.corr()


# In[8]:


df.isnull().sum()


# In[9]:



df.duplicated().sum()


# In[10]:


for i in df.columns:
    print(i,len(df[i].unique()))


# In[11]:


df.hist(figsize=(35,40))
plt.show()


# In[12]:


sns.set(rc={'figure.figsize':(25,40)})
sns.countplot(x=df['iyear'], hue=df['region_txt'])
plt.title('Region-wise terrorist activity in each year ')


# In[13]:


pd.crosstab(df['iyear'],df['region_txt']).plot(kind='line',figsize=(15,20))
plt.title('Region-wise terrorist activity in each year ')
plt.ylabel('Number of Attacks')
plt.xlabel('years')
plt.show()

print('Middle east & North Africa has been reported to have more incidents')


# In[14]:


# Most effected region
df['region_txt'].value_counts()


# In[15]:


# Most effected countries
df['country_txt'].value_counts()


# In[16]:


#value_counts() function returns object containing counts of unique values.
#The resulting object will be in descending order 
#so that the first element is the most frequently-occurring element.


# In[17]:


plt.figure(figsize=(15,6))
sns.barplot(df['country_txt'].value_counts()[:10].index,df['country_txt'].value_counts()[:10].values)
plt.title('Top 10 countries effested by Attacks',fontsize=15)
plt.ylabel('Counts',fontsize=15)
plt.xlabel('Countries',fontsize=15)
plt.show()

print('Most effected country is Iraq & mostly belongs to Asia continent')


# In[18]:


plt.figure(figsize=(15,6))
sns.barplot(df['city'].value_counts()[:10].index,df['city'].value_counts()[:10].values)
plt.title('Top 10 city effested by Attacks',fontsize=15)
plt.ylabel('Counts',fontsize=15)
plt.xlabel('City',fontsize=15)
plt.show()

print('Most effected city is baghdad & unknown')


# In[19]:


sns.set(rc={'figure.figsize':(30,7)})
sns.countplot(x=df['region_txt'], hue=df['weaptype1_txt'])
plt.title('Weapon type used in per region ')
plt.xlabel('Region',fontsize=15)
print('Explosives, chemicals and Firearms are widely used as a weapon type')


# In[20]:


sns.set(rc={'figure.figsize':(30,7)})
sns.countplot(x=df['region_txt'], hue=df['attacktype1_txt'])
plt.title('Attack type used in per region ')
plt.xlabel('Region',fontsize=15)
print('Bombing/Explosion is highly used attack type and region is Middle East/North Africa')


# In[21]:


sns.set(rc={'figure.figsize':(35,10)})
sns.countplot(x=df['region_txt'], hue=df['targtype1_txt'])
plt.title('Attack type used in per region ')
plt.xlabel('Region',fontsize=15)
print('Private Citizens/Property is the highly target type in each region')


# In[22]:



df.plot(kind='scatter',x='nkill',y='nwound',color='green',figsize=(10,10),fontsize=15)
plt.xlabel('Kill')
plt.ylabel('Wound')
plt.title('kill - wound Scatter plot')
plt.show()


# In[23]:


# Now we are going to analysis the data of INDIA

new=df[df['country_txt']=='India']['city']
new.value_counts()
# Most effected city of india - Sri nagar 


# In[24]:


plt.figure(figsize=(10,5))
sns.barplot(new.value_counts()[:10].index,new.value_counts()[:10].values)
plt.title('Top 10 Terrorism effected city of india')
plt.xlabel('City', fontsize=15)
plt.ylabel('Attacks', fontsize=15)
plt.show()


# In[25]:


new=df[df['country_txt']=='India'][['city','iyear','attacktype1_txt','gname']]
new[:10]
# This is the top 10 location of india with attack details.


# In[26]:


df['iyear'].value_counts()[:10]


# In[27]:


#top 10 terriorst groups 


df['gname'].value_counts()[:10]


# In[28]:


new=df[df['country_txt']=='India']['gname']
new.value_counts()
# This is the top terriorst groups of india


# In[29]:



new=df[df['country_txt']=='India']['iyear']
new.value_counts()
# This is the top terriorst attacks of india


# In[30]:


new=df[df['country_txt']=='India']['attacktype1_txt']
new.value_counts()
# This is the top terriorst attack type


# # Conclusion :
# ##### 1. Most attacks in Country - Iraq
# ##### 2. Most attacks in City - Baghdad
# ##### 3. Most attacks in Region - Middle East & North Africa
# ##### 4. Least attacks in Region - Australasia & Oceania
# ##### 5. Most attacks in year - 2014
# ##### 6. Most attacks in Month - 5
# ##### 7. Most attacks by Group - Taliban
# ##### 8. Most attacks type - Bombing/Explsion
# ##### 9. Most target type by terriorst -  Private Citizens/Property 
# ##### 10. Most target Continent - Asia
# ##### 11. Least attacks type - Hijecking
# ##### 12. least attacks in year - 1971 
# 
# 
# ### India:
# 
#      1. Most effected state - J&K
#      2. Most effected city - Sri Nagar
#      3. Most terriorst group - CPI- Maoist
#      4. Most effected yesr - 2016
#      5. Least effected year - 1976,1977,1972,1975
#      6. Most attack type - Bombing/Explosion
#      7. Least attack type - Hijecking
#      
#   In the majority of acts of terrorism, the majority rate and injuries were low but a small number of action led to too many deaths and injuries.
#      
# 
# 
# 
# 
# Note: I have not considered unknown data.

# In[ ]:




