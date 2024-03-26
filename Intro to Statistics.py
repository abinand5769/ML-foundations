#!/usr/bin/env python
# coding: utf-8

# # Segment 1: Frequentist Statistics

# In[1]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


np.random.seed(42)


# Measures of Central Tendency
# Measures of central tendency provide a summary statistic on the center of a given distribution, a.k.a., the "average" value of the distribution.

# In[3]:


x = st.skewnorm.rvs(10, size=1000)


# In[4]:


x[0:20]


# In[5]:


fig, ax = plt.subplots()
_ = plt.hist(x, color = 'lightgray')


# Mean
# The most common measure of central tendency, synonomous with the term "average", is the mean,

# In[6]:


xbar = x.mean()
xbar


# In[7]:


fig, ax = plt.subplots()
plt.axvline(x = x.mean(), color='orange')
_ = plt.hist(x, color = 'lightgray')


# Median
# The second most common measure of central tendency is the median, the midpoint value in the distribution:

# In[8]:


np.median(x) 

Measures of Dispersion
# In[10]:


x.var()  #variance


# In[11]:


sigma = x.std()
sigma  # standard deviation


# # Gaussian Distribution

# 
# After Carl Friedrich Gauss. Also known as normal distribution:

# In[12]:


x = np.random.normal(size=10000)


# In[13]:


sns.set_style('ticks')


# In[15]:


sns.displot(x, kde=True)


# In[16]:


x.mean()


# In[17]:


x.std()


# ...it is a standard normal distribution (a.k.a., standard Gaussian distribution or ***z*-distribution**), which can be denoted as 
#  (noting that 
#  here because 
# ).
# 
# Normal distributions are by far the most common distribution in statistics and machine learning. They are typically the default option, particularly if you have limited information about the random process you're modeling, because:
# 
# Normal distributions assume the greatest possible uncertainty about the random variable they represent (relative to any other distribution of equivalent variance). Details of this are beyond the scope of this tutorial.
# Simple and very complex random processes alike are, under all common conditions, normally distributed when we sample values from the process. Since we sample data for statistical and machine learning models alike, this so-called central limit theorem (covered next) is a critically important concept.

# # The Central Limit Theorem

# In[18]:


x_sample = np.random.choice(x, size=10, replace=False)
x_sample


# In[19]:


x_sample.mean()


# In[20]:


def sample_mean_calculator(input_dist, sample_size, n_samples):
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(input_dist, size=sample_size, replace=False)
        sample_means.append(sample.mean())
    return sample_means


# In[21]:


sns.displot(sample_mean_calculator(x, 10, 10), color='green', kde=True)


# In[23]:


sns.displot(sample_mean_calculator(x, 10, 1000), color='green', kde=True)


# In[24]:


sns.displot(sample_mean_calculator(x, 1000, 1000), color='green', kde=True)


# Sampling from a skewed distribution

# In[25]:


s = st.skewnorm.rvs(10, size=10000)


# In[26]:


sns.displot(s, kde=True)


# In[27]:


sns.displot(sample_mean_calculator(s, 10, 1000), color='green', kde=True)


# In[28]:


sns.displot(sample_mean_calculator(s, 1000, 1000), color='green', kde=True)


# 
# Sampling from a multimodal distribution

# In[29]:


m = np.concatenate((np.random.normal(size=5000), np.random.normal(loc = 4.0, size=5000)))


# In[30]:


sns.displot(m, kde=True)


# In[31]:


sns.displot(sample_mean_calculator(m, 1000, 1000), color='green', kde=True)


# Sampling from uniform

# In[32]:


u = np.random.uniform(size=10000)


# In[33]:


sns.displot(u)


# In[34]:


sns.displot(sample_mean_calculator(u, 1000, 1000), color='green', kde=True)


# Therefore, with large enough sample sizes, we can assume the sampling distribution of the means will be normally distributed, allowing us to apply statistical and ML models that are configured for normally distributed noise, which is often the default assumption.
# 
# As an example, the "t-test" (covered shortly in Intro to Stats) allows us to infer whether two samples come from different populations (say, an experimental group that receives a treatment and a control group that receives a placebo). Thanks to the CLT, we can use this test even if we have no idea what the underlying distributions of the populations being tested are, which may be the case more frequently than not.

# z-scores

# In[35]:


x_i = 85
mu = 60
sigma = 10


# In[36]:


x = np.random.normal(mu, sigma, 10000)


# In[37]:


sns.displot(x, color='gray')
ax.set_xlim(0, 100)
plt.axvline(mu, color='orange')
for v in [-3, -2, -1, 1, 2, 3]:
    plt.axvline(mu+v*sigma, color='olivedrab')
_ = plt.axvline(x_i, color='purple')


# In[38]:


z = (x_i - mu)/sigma
z


# In[39]:


z = (x_i - np.mean(x))/np.std(x)
z


# In[40]:


len(np.where(x > 85)[0])


# In[41]:


100*69/10000


# In[42]:


np.percentile(x, 99)


# In[43]:


mu = 90
sigma = 2


# In[44]:


y = np.random.normal(mu, sigma, 10000)


# In[45]:


sns.displot(y, color='gray')
plt.axvline(mu, color='orange')
for v in [-3, -2, -1, 1, 2, 3]:
    plt.axvline(mu+v*sigma, color='olivedrab')
_ = plt.axvline(x_i, color='purple')


# In[46]:


z = (x_i - mu)/sigma
z


# In[47]:


z = (x_i - np.mean(y))/np.std(y)
z


# In[48]:


len(np.where(y > 85)[0])


# In[49]:


100*9933/10000


# In[50]:


10000-9933


# In[51]:


np.percentile(y, 1)


# A frequentist convention is to consider a data point that lies further than three standard deviations from the mean to be an outlier.
# 
# It's a good idea to individually investigate outliers in your data as they may represent an erroneous data point (e.g., some data by accident, a data-entry error, or a failed experiment) that perhaps should be removed from further analysis (especially, as outliers can have an outsized impact on statistics including mean and correlation). It may even tip you off to a major issue with your data-collection methodology or your ML model that can be resolved or that you could have a unit test for.

# Confidence Intervals

# When examining sample means as we have been for the t-test, a useful statistical tool is the confidence interval (CI), which we for example often see associated with polling results when there's an upcoming election. CIs allow us to make statements such as "there is a 95% chance that the population mean lies within this particular range of values".
# 
# We can calculate a CI by rearranging the z-score formula:
# 

# In[87]:


x = np.array([48, 50, 54, 60, 49, 55, 59, 62])


# In[88]:


xbar = x.mean()
s = x.std()
n = x.size


# In[89]:


z = 1.96


# In[90]:


def CIerr_calc(my_z, my_s, my_n):
    return my_z*(my_s/my_n**(1/2))


# In[91]:


CIerr = CIerr_calc(z, s, n)


# In[92]:


CIerr


# In[94]:


xbar + CIerr


# In[95]:


xbar - CIerr


# Therefore, there's a 95% chance that the true mean yield of our GMO yeast lies in the range of 51.2 to 58.1 liters. Since this CI doesn't overlap with the established baseline mean of 50L, this corresponds to stating that the GMO yield is significantly greater than the baseline where 
# , as we already determined:

# Pearson Correlation Coefficient

# If we have two vectors of the same length, X
#  and Y
# , where each element of X
#  is paired with the corresponding element of Y
# , covariance provides a measure of how related the variables are to each other:
#  
# 
# A drawback of covariance is that it confounds the relative scale of two variables with a measure of the variables' relatedness. Correlation builds on covariance and overcomes this drawback via rescaling, thereby measuring (linear) relatedness exclusively. Correlation is much more common because of this difference.

# In[99]:


iris = sns.load_dataset('iris')
iris


# In[100]:


x = iris.sepal_length
y = iris.petal_length


# In[101]:


sns.set_style('darkgrid')


# In[102]:


sns.scatterplot(x=x, y=y)


# In[103]:


n = iris.sepal_width.size


# In[104]:


xbar, ybar = x.mean(), y.mean()


# In[105]:


product = []
for i in range(n):
    product.append((x[i]-xbar)*(y[i]-ybar))


# In[106]:


cov = sum(product)/n
cov


# In[107]:


r = cov/(np.std(x)*np.std(y))
r


# In[108]:


t = r*((n-2)/(1-r**2))**(1/2)
t


# In[109]:


p = p_from_t(t, n-1) 
p


# In[110]:


-np.log10(p)


# In[111]:


#This confirms that iris sepal length is extremely positively correlated with petal length.

#All of the above can be done in a single line with SciPy's pearsonr() method:


# In[112]:


st.pearsonr(x, y)


# In[113]:


sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width)


# In[114]:


st.pearsonr(iris.sepal_length, iris.sepal_width)


# 
# The Coefficient of Determination

# In[115]:


rsq = r**2
rsq


# In this case, it indicates that 76% of the variance in iris petal length can be explained by sepal length. (This is easier to understand where one variable could straightforwardly drive variation in the other; more on that in Segment 2.)

# In[116]:


#For comparison, only 1.4% of the variance in sepal width can be explained by sepal length:


# In[117]:


st.pearsonr(iris.sepal_length, iris.sepal_width)[0]**2


# In[ ]:




