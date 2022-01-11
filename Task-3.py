#!/usr/bin/env python
# coding: utf-8

# # # The Sparks Foundation - Data Science & Business Analytics
# 
# ### Perform ‘Exploratory Data Analysis’ on dataset ‘SampleSuperstore’
# 
# #### In this task we will try to find out the weak areas where we can work to make more profit.
# ### Data Set can be found on Dataset: https://bit.ly/3i4rbWl

# In[ ]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

# To ignore the warnings 
import warnings as wg


# In[2]:



df = pd.read_csv("SampleSuperstore.csv")

df.head()


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


df.columns


# In[7]:


for i in df.columns:
    print(i,len(df[i].unique()))


# In[9]:


df.isnull().sum()


# In[11]:


df.describe()


# In[14]:


df.duplicated().sum()


# In[15]:


df.drop_duplicates()
df.head()


# In[17]:


dataset=df.drop(['Postal Code'],axis=1)
dataset.head()


# # Data Visualization

# In[19]:


sns.pairplot(dataset,hue="Category")


# In[20]:


sns.pairplot(dataset,hue="Region")


# In[21]:


dataset.corr()


# In[22]:


plt.figure(figsize=(8,4))
sns.heatmap(dataset.corr(), annot=True)
plt.show()


# # observtion
# ### Discount and profit are negatively related
# ### Quantity and profit are moderately related
# ### Sales and profit are moderately realted

# In[27]:


plt.figure(figsize = (4,4))
plt.title('Category')
plt.pie(dataset['Category'].value_counts(), labels=dataset['Category'].value_counts().index,autopct='%.2f%%')
plt.show()


# In[28]:


plt.figure(figsize = (4,4))
plt.title('Region')
plt.pie(dataset['Region'].value_counts(), labels=dataset['Region'].value_counts().index,autopct='%.2f%%')
plt.show()


# In[29]:



plt.figure(figsize = (4,4))
plt.title('Segments')
plt.pie(dataset['Segment'].value_counts(), labels=dataset['Segment'].value_counts().index,autopct='%.2f%%')
plt.show()


# In[30]:


# Count plot of ship mode of distribution
plt.figure(figsize = (4,4))
plt.title('Ship Mode')
plt.pie(dataset['Ship Mode'].value_counts(), labels=dataset['Ship Mode'].value_counts().index,autopct='%.2f%%')
plt.show()


# # Segment wise analysis of profit,sales,discount

# In[32]:


dataset.groupby('Segment').mean()


# In[33]:


dataset.groupby(['Segment'])[['Sales', 'Discount', 'Profit']].mean()


# # Quantity wise analysis of profit,sales,discount

# In[34]:


dataset.groupby(['Quantity'])[['Sales', 'Discount', 'Profit']].mean()


# In[35]:


sns.set(rc={'figure.figsize':(10,8)})
sns.countplot(x=dataset['Quantity'], hue=dataset['Region'])
plt.title('Region-wise ordered Quantity')


# In[36]:


sns.set(rc={'figure.figsize':(10,8)})
sns.countplot(x=dataset['Region'], hue=dataset['Category'])
plt.title('Region-wise ordered product categories')


# # observation
# # West region has ordered more,south region has ordered less

# #  Category wise analysis of profit,sales,discount 

# In[37]:


dataset.groupby(['Category'])[['Sales', 'Discount', 'Profit']].mean()


# In[38]:


s = dataset.groupby("Category").Sales.sum()

# computing top categories in terms of profit 
p = dataset.groupby("Category").Profit.sum()

# plotting to see it visually
plt.style.use('seaborn')
s.plot(kind = 'bar',figsize = (10,5),fontsize = 14, color='red')
p.plot(kind = 'bar',figsize = (10,5),fontsize = 14,color='yellow')
plt.xlabel('Category',fontsize = 15)
plt.ylabel('Total Sales/Profits',fontsize = 15)
plt.title("Category Sales vs Profit",fontsize = 15)
plt.show()


# # Sub Category wise analysis of profit,sales,discount 

# In[39]:


dataset.groupby(['Sub-Category'])[['Sales', 'Discount', 'Profit']].mean()


# In[40]:


plt.figure(figsize=(18,8))
sns.countplot(df['Sub-Category'])
plt.title('Sub-Category',fontsize=15)


# In[41]:


# computing top sub-categories in terms of sales

sc = dataset.groupby("Sub-Category").Sales.sum()

# computing top sub-categories in terms of profit
scp = dataset.groupby("Sub-Category").Profit.sum()

# plotting to see it visually
plt.style.use('seaborn')
sc.plot(kind = 'bar',figsize = (10,5),fontsize = 14,color = 'black')
scp.plot(kind = 'bar',figsize = (10,5),fontsize = 14, color = 'y')
plt.xlabel('Sub-Category',fontsize = 15)
plt.ylabel('Total Sales/Profits',fontsize = 15)
plt.title("Sub-Category Sales vs Profit",fontsize = 15)
plt.show()


# In[44]:


# A more detailed view
plt.figure(figsize=(12,12))
statewise = dataset.groupby(['Sub-Category'])['Profit'].sum()
statewise.plot.barh() 
# h for horizontal


# # The above graph clearly shows that Copiers and Phones have the highest sales and profit and tables has negative profit

# In[45]:


plt.figure(figsize=(10,4))
sns.countplot(df['Discount'])
plt.title('Discount Graph with respect of count (Quantity)',fontsize=15)


# In[46]:


fig,axes = plt.subplots(2,2,figsize=(14,12))
fig.suptitle("Distribution plots", fontsize=16)
sns.distplot(df['Sales'],ax=axes[0,0])
sns.distplot(df['Profit'],ax=axes[0,1])
sns.distplot(df['Discount'],ax=axes[1,0])
sns.distplot(df['Quantity'],ax=axes[1,1])
axes[0][0].set_title('Sales Distribution')
axes[0][1].set_title('Profit Distribution')
axes[1][0].set_title('Discount Distribution')
axes[1][1].set_title('Quantity Distribution')
plt.show()


# # Shows the distrubition sales,profit,discount,quantity

# ### In the above graph, Sales distribution is rightly skewed and profit distribution is seems normal skewed.

# In[48]:


# State- wise Analysis
new_df=dataset['State'].value_counts()
new_df.head()


# In[49]:


min=dataset['State'].value_counts().min()
min


# In[50]:


max=dataset['State'].value_counts().max()
max


#  ### State - Wise analysis of Profit, Sales & Discount

# In[51]:


dataset.groupby(['State'])[['Sales', 'Discount', 'Profit']].mean()


# In[52]:


new_df.plot(kind='bar',figsize=(12,6))
plt.ylabel('Number of Deals')
plt.xlabel('States')
plt.title('states-wise Dealings')


# In[53]:


# computing states in terms of sales 
ss = dataset.groupby("State").Sales.sum()

# computing states in terms of profit 
ps = dataset.groupby("State").Profit.sum()

plt.style.use('seaborn')
ss.plot(kind = 'bar',figsize = (14,6),fontsize = 14,color='g')
ps.plot(kind = 'bar',figsize = (14,6),fontsize = 14, color = 'k')
plt.xlabel('States',fontsize = 15)
plt.ylabel('Total sales',fontsize = 15)
plt.title("States Sales vs Profit",fontsize = 15)
plt.show()


# #  Conclusion :
# ## 1. Weak Areas - 
# ###### i) We should limit sales of furniture and increase that of technology and office suppliers as furniture has very less profit as compared to sales.
# ###### ii) Increase sales more in the east as profit is more. 
# ###### iii) Need to give some discount and other offer in state (colorado, texas, north carolina,ohio) because there is sale is fine but no profit.
# 
# ## 2. Other Findings :
# ###### i) Over Less quantity of products also the sales were high.
# ###### ii) The features Profit and Discounts are highly related.
# ###### iii) The Home Office provides highest sales followed by Corporate by a slight variation.
# ###### iv) We should concentrate on the states like 'New York' and 'California' to make more profits.

# In[ ]:




