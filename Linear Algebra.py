#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra II: Matrix Operations

# In[2]:


import numpy as np


# In[3]:


x=np.array([25,2,3])
x


# In[4]:


x.shape


# In[5]:


x = np.array([[25, 2, 5]])


# In[6]:


x


# In[7]:


x.shape


# In[8]:


x.T


# In[9]:


x.T.shape


# In[10]:


X = np.array([[25, 2], [5, 26], [3, 7]])
X


# In[11]:


X.shape


# In[12]:


X.shape


# In[ ]:





# Matrices

# In[13]:


X = np.array([[25, 2], [5, 26], [3, 7]])
X


# In[14]:


X.shape


# Matrix Transposition

# In[16]:


X


# In[18]:


X.T


# Matrix Multiplication

# In[19]:


X*3


# Using the multiplication operator on two tensors of the same size in PyTorch (or Numpy or TensorFlow) applies element-wise operations. This is the Hadamard product

# In[20]:


A = np.array([[3, 4], [5, 6], [7, 8]])
A


# In[21]:


X


# In[22]:


X * A


# # Principal Component Analysis

# In[23]:


from sklearn import datasets
iris = datasets.load_iris()


# In[24]:


iris.data.shape


# In[25]:


iris.get("feature_names")


# In[26]:


iris.data[0:6,:]


# In[27]:


from sklearn.decomposition import PCA


# In[28]:


pca = PCA(n_components=2)


# In[29]:


X = pca.fit_transform(iris.data)


# In[30]:


X.shape


# In[31]:


X[0:6,:]


# In[32]:


iris.target.shape


# In[33]:


iris.target[0:6]


# In[34]:


unique_elements, counts_elements = np.unique(iris.target, return_counts=True)
np.asarray((unique_elements, counts_elements))


# In[35]:


list(iris.target_names)


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


plt.scatter(X[:, 0], X[:, 1], c=iris.target)


# In[ ]:




