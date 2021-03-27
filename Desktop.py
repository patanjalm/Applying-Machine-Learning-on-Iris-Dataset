#!/usr/bin/env python
# coding: utf-8

# # Data set Information
# ### Data set contain 3 clasess of 50 instance each,where each class refers to type of iris plant.One class is linearly separable from each other.
# ### Attribute information:
# 1.Sepal length in cm 
# 
# 2.Sepal width in cm
# 
# 3.Petal length in cm
# 
# 4.Petal width in cm
# 
# class 1.Iris Setosa,2.Iris Versicolour,3.Iris Virginica

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# # Load the Dataset

# In[2]:


df = pd.read_csv(r"D:\project ....machine learning dataset of iris\archive (1)\Iris.csv")


# In[3]:


df.head()


# In[4]:


# delete a column
df=df.drop(columns=['Id'])


# In[5]:


df.head()


# In[6]:


# To see stats into the data
df.describe()


# In[7]:


# basic info of dataset
df.info()


# In[8]:


# To display number of sample on each class
df['Species'].value_counts()


# # Preprocessing Data Analysis

# In[9]:


# Check for null value
df.isnull().sum()


# # Exploratory Data Analysis

# In[10]:


# histogram
df['SepalLengthCm'].hist()


# In[11]:


df['SepalWidthCm'].hist()


# In[12]:


df['PetalLengthCm'].hist()


# In[13]:


df['PetalWidthCm'].hist()


# In[14]:


# scatterplot 
colors = ['red','orange','blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[15]:


df.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm');
plt.show()


# In[16]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Species",size=4).map(plt.scatter,"PetalLengthCm","PetalWidthCm").add_legend();
plt.show();


# In[17]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Species",size=4).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend();
plt.show();


# In[18]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Species",size=4).map(plt.scatter,"SepalWidthCm","PetalWidthCm").add_legend();
plt.show();


# # Correlation matrix

# ### A correlation matrix is a table showing correlation coefficient between variables.Each cell in the table shows the correlation between two variables.The value is in range of -1 to 1.If two variable have high correlation we neglect one variable from those two.

# In[19]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax)


# # Label Encoder

# ### In machine learning,we usually deal with datasets which contain multiple lables in one or more than one columns.These labels can be in form of words and numbers.Label encoding refers to converting the labels into numbric form so as to convert it into machine readable form.

# In[20]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[21]:


df['Species']=le.fit_transform(df["Species"])
df.head()


# # Modle Tranning

# In[22]:


from sklearn.model_selection import train_test_split
x= df.drop(columns=['Species'])
y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[23]:


# Logistic Regration
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[24]:


model.fit(x_train,y_train)


# In[27]:


model.score(x_train,y_train)


# In[ ]:




