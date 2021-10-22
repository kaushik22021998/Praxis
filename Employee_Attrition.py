#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pickle
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


data = pd.read_csv('HR_comma_sep.csv')


# In[3]:


data.drop(['Department','salary'],axis=1,inplace=True)


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data['left'].value_counts(normalize=True)


# In[9]:


data.hist(bins=30,figsize=(25,15))


# In[10]:


plt.hist(data["time_spend_company"])
plt.xlabel("Time Spent at the company")
plt.ylabel("Frequency")


# In[11]:


plt.hist(data['satisfaction_level'])
plt.xlabel('Satisfaction Level')
plt.ylabel('Frequency')


# In[12]:


plt.hist(data["average_montly_hours"],bins=50)
plt.xlabel("Average Monthly Hours")
plt.ylabel("Frequency")


# In[13]:


plt.hist(data["last_evaluation"],bins=50)
plt.xlabel("Evaluation Score")
plt.ylabel("Frequency")


# In[14]:


sns.boxplot(data["last_evaluation"])


# In[15]:


sns.boxplot(data['satisfaction_level'])


# In[16]:


sns.boxplot(data['average_montly_hours'])


# In[17]:


sns.boxplot(data['number_project'])


# In[18]:


sns.boxplot(data['time_spend_company'])


# In[19]:


df_numeric = data.iloc[:,:5]
fig, ax = plt.subplots(figsize = [20,6])
sns.heatmap(df_numeric.corr(), annot= True, cmap = "Blues");


# In[20]:


df_numeric.corr()


# #***Z-test - One sample mean***

# In[21]:


sample = data.sample(n=50,replace=False,random_state = 0)
sample.head()


# In[22]:


sample_mean = sample['average_montly_hours'].mean()
sample_mean


# In[23]:


pop_std = data['average_montly_hours'].std()
pop_std


# Z-test 
# *   H0 : Mean <= 160
# *   H0 : Mean > 160
# 
# This is one tailed test with 95% confidence

# In[24]:


z_cal = (sample_mean - 160)/((pop_std/np.sqrt(sample.shape[0])))
z_cal


# 
# *   Calculated Z value = 5.38
# *   Tabulated Z value = 1.645
# 
# Calculated value is greater than actual value. H1 is true. 
# 
# 

# #***Z test - Two Sample Means***

# In[25]:


pop1 = data[data['left']==0]
pop1.shape


# In[26]:


pop2 = data[data['left']==1]
pop2.shape


# In[27]:


pop1_mean = pop1['average_montly_hours'].mean()
pop1_std = pop1['average_montly_hours'].std()
pop2_mean = pop2['average_montly_hours'].mean()
pop2_std = pop2['average_montly_hours'].std()


# In[28]:


pop1_mean


# In[29]:


pop2_mean


# In[30]:


pop1_std


# In[31]:


pop2_std


# In[32]:


ste1 = pop1_std**2/pop1.shape[0]
ste2 = pop2_std**2/pop2.shape[0]
z_score_2 = (pop1_mean - pop2_mean)/np.sqrt(ste1+ste2)
z_score_2


# H0 : There is no significant difference in the average monthly working hours of people who left the organisation and people who didn't leave the organisation
# 
# H1 : There is a significant difference in the average monthly working hours of people who left the organisation and people who didn't leave the organisation
# 
# Two tailed test with 95% confidence
# 
# Calculated Value is 7.53
# 
# Tabulated Value is 1.96
# 
# Calculated Value is greater than Tabulated Value
# 
# H1 is true
# 
# 

# In[33]:


pop3 = data[data['promotion_last_5years']==0]
pop3.shape


# In[34]:


pop4 = data[data['promotion_last_5years']==1]
pop4.shape


# In[35]:


pop3_mean = pop3['satisfaction_level'].mean()
pop3_std = pop3['satisfaction_level'].std()
pop4_mean = pop4['satisfaction_level'].mean()
pop4_std = pop4['satisfaction_level'].std()


# In[36]:


pop3_mean


# In[37]:


pop4_mean


# In[38]:


pop3_std


# In[39]:


pop4_std


# In[40]:


ste3 = pop3_std**2/pop3.shape[0]
ste4 = pop4_std**2/pop4.shape[0]
z_score_3 = (pop3_mean - pop4_mean)/np.sqrt(ste3+ste4)
z_score_3


# H0 : There is no significant difference in the satisfaction level of people who got promoted in the last 5 years and people who didn't promoted in the last 5 years
# 
# H1 : There is a significant difference in the satisfaction level of people who got promoted in the last 5 years and people who didn't promoted in the last 5 years
# 
# Calculated Value is 3.65
# 
# Tabulated Value is 1.96
# 
# Calculated Value is greater than than Tabulated Value
# 
# H1 is true
# 
# 

# In[41]:


pop5_mean = pop3['number_project'].mean()
pop5_std = pop3['number_project'].std()
pop6_mean = pop4['number_project'].mean()
pop6_std = pop4['number_project'].std()


# In[42]:


pop5_mean


# In[43]:


pop5_std


# In[44]:


pop6_mean


# In[45]:


pop6_std


# In[46]:


ste5 = pop5_std**2/pop3.shape[0]
ste6 = pop6_std**2/pop4.shape[0]
z_score_4 = (pop5_mean - pop6_mean)/np.sqrt(ste5+ste6)
z_score_4


# H0 : There is a no significant difference in the number of projects handled by people who got promoted and who didn't get promoted
# 
# H1 : There is a significant difference in the number of projects handled by people who got promoted and who didn't get promoted
# 
# Two tailed test with 95 % confidence
# 
# Calculated Value is 0.86
# 
# Tabulated Value is 1.96
# 
# Calculated Value is less than than Tabulated Value
# 
# H0 is true

# #**F-test for two samples**

# In[47]:


pop7 = data[data['promotion_last_5years']==0]
pop7.shape


# In[48]:


pop8 = data[data['promotion_last_5years']==1]
pop8.shape


# In[49]:


pop7_mean = pop7['average_montly_hours'].mean()
pop7_std = pop7['average_montly_hours'].std()
pop8_mean = pop8['average_montly_hours'].mean()
pop8_std = pop8['average_montly_hours'].std()


# In[50]:


pop7_std


# In[51]:


pop8_std


# In[52]:


sample_7 = pop7.sample(n=50,random_state=2)
sample_7.shape


# In[53]:


sample_8 = pop8.sample(n=50,random_state=8)
sample_8.shape


# In[54]:


sample7_std = sample_7['average_montly_hours'].std()
sample7_std


# In[55]:


sample8_std = sample_8['average_montly_hours'].std()
sample8_std 


# In[56]:


F_stat = sample7_std**2/sample8_std**2
F_stat


# Ho: No difference in variances of average monthly hours of people who got promoted and who didn't get promoted
# 
# Ha: There is a significant difference in variances of average monthly hours of people who got promoted and who didn't get promoted
# 
# For DOF(49,49) at alpha = 0.025, Critical F value = 1.76
# 
# Calculated F value = 1.02
# 
# Calculated F < Critical F
# 
# Ho is true
# 
# 

# #***Logistic Regression***

# In[59]:


logit_data = data
logit_data.shape


# In[61]:


model_data = logit_data.copy()
model_data.shape


# In[62]:


from sklearn.model_selection import train_test_split
target = model_data['left']
features = model_data.drop('left',axis=1)
x = features[:]
y = target[:]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=123,stratify=y)


# In[65]:


log_reg = sm.Logit(y_train, x_train).fit()


# In[66]:


print(log_reg.summary())


# In[67]:


# performing predictions on the test datdaset
yhat = log_reg.predict(x_test)
prediction = list(map(round, yhat))
 
# comparing original and predicted values of y
print('Actual values', list(y_test.values))
print('Predictions :', prediction)


# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# In[69]:


lr=LogisticRegression(class_weight = "balanced",max_iter=1000)
lr.fit(x_train,y_train)


# In[70]:


# Saving model to disk
pickle.dump(lr, open('modellr.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modellr.pkl','rb'))
print(model.predict([[0.5, 0.5,2,220,5,0,1]]))

# In[ ]:




