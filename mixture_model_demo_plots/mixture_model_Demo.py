#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import lognormal, normal


# In[64]:


from scipy.stats import lognorm, norm
import pylab


# In[24]:


sns.set()


# In[75]:


mu, sigma = 3,.5;
num=5000
s1= lognormal( mu, sigma,5*num);
s2=normal(20*mu, 6*sigma, 2*num)
s3=normal(15*mu, 8*sigma, 3*num)
s4=normal(10*mu, 3*sigma, 1*num)
s5=normal(25*mu, 16*sigma, 2*num)

s = np.concatenate((s1,s2,s3,s4,s5))
plt.figure()
count, bins, _ = plt.hist(s,100, density=True)
plt.savefig('mixture_model_raw.png')
plt.show()


x=np.linspace(min(bins), max(bins), 5000)
def lognorm_pdf(x, mu, sigma):
    return(np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))

def norm_pdf(x, mu, sigma):
    return(np.exp(-(x - mu)**2 / (2 * sigma**2))/ (sigma * np.sqrt(2 * np.pi)))

plt.figure()
count, bins, _ = plt.hist(s,100, density=True)
pylab.plot(x, lognorm_pdf(x, mu, sigma)*5/14 ,'g--', linewidth=2,  label='lognormal')
pylab.plot(x, norm_pdf(x, 20*mu, 6*sigma)*2/14,'r--', linewidth=2, label='normal')
pylab.plot(x, norm_pdf(x, 15*mu, 8*sigma)*3/14, 'r--',linewidth=2)
pylab.plot(x, norm_pdf(x, 10*mu, 3*sigma)/14,'r--', linewidth=2 )
pylab.plot(x, norm_pdf(x, 25*mu, 16*sigma)*2/14,'r--', linewidth=2)
pylab.legend(loc='upper right')
plt.savefig('mixture_model_fitted.png')
plt.show()


# In[30]:


type(s)


# In[63]:


import pylab


# In[ ]:




