import datetime
import warnings

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn import linear_model
import matplotlib.ticker

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd

# get_ipython().run_line_magic('matplotlib', 'notebook')
# Linear Regression
# used for 3D plot
# igore warning
warnings.filterwarnings("ignore")

# # Reading data and preprocessing

# In[2]:

df = pd.read_csv('swapLiborData.csv')

# convert number to datatime format
for i in range(df.shape[0]):
    df.loc[i, 'Date'] = pd.to_datetime(
        '1899-12-30') + pd.to_timedelta(df.loc[i, 'Date'], 'D')

df.head(5)

# # regress 5-yr against 2-yr

# In[3]:

len1 = len(df)

t1 = 0
t2 = int(np.ceil(len1 / 2))

xX = df.iloc[t1:t2, 6:7]
yY = df.iloc[t1:t2, 8:9]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 6:7] @ B.T
# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 8:9])

# adjust display setting
# plt.figure(figsize=(8,5)) # set the figure size
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('5-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 5-yr', 'original 5-yr'], loc='best')

plt.show()

# # regress 5-yr against 2-yr

# In[7]:

len1 = len(df)

t1 = int(np.ceil(len1 / 2))
t2 = len1

xX = df.iloc[t1:t2, 6:7]
yY = df.iloc[t1:t2, 8:9]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 6:7] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 8:9])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('5-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 5-yr', 'original 5-yr'], loc='best')

plt.show()

# # regress 5-yr against 2-yr

# In[8]:

len1 = len(df)

t1 = 0
t2 = len1

xX = df.iloc[t1:t2, 6:7]
yY = df.iloc[t1:t2, 8:9]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 6:7] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 8:9])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('5-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 5-yr', 'original 5-yr'], loc='best')

plt.show()

# # regress 30-yr against 15-yr

# In[9]:

len1 = len(df)

t1 = 0
t2 = int(np.ceil(len1 / 2))

xX = df.iloc[t1:t2, 11:12]
yY = df.iloc[t1:t2, 12:13]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 11:12] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 12:13])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('30-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 30-yr', 'original 30-yr'], loc='best')

plt.show()

# # regress 30-yr against 15-yr

# In[10]:

len1 = len(df)

t1 = int(np.ceil(len1 / 2))
t2 = len1

xX = df.iloc[t1:t2, 11:12]
yY = df.iloc[t1:t2, 12:13]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 11:12] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 12:13])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('30-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 30-yr', 'original 30-yr'], loc='best')

plt.show()

# # regress 30-yr against 15-yr

# In[11]:

len1 = len(df)

t1 = 0
t2 = len1

xX = df.iloc[t1:t2, 11:12]
yY = df.iloc[t1:t2, 12:13]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, 11:12] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 12:13])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('30-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 30-yr', 'original 30-yr'], loc='best')

plt.show()

# # regress 30-yr against 2-yr, 5-yr, and 10-yr

# In[12]:

len1 = len(df)

t1 = 0
t2 = int(np.ceil(len1 / 2))

xX = df.iloc[t1:t2, [6, 8, 10]]
yY = df.iloc[t1:t2, 12:13]
regr = linear_model.LinearRegression()
regr.fit(xX, yY)
B = regr.coef_
b_0 = regr.intercept_
rSquared = regr.score(xX, yY)

print(b_0, B)
print(rSquared)

yHat = b_0 + df.iloc[t1:t2, [6, 8, 10]] @ B.T

# plot data
plt.figure(figsize=(8, 5))  # set the figure size

# plt.plot(df.iloc[t1:t2, 0], yHat)
plt.plot(df.iloc[t1:t2, 0], df.iloc[t1:t2, 12:13])

# adjust display setting
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation='horizontal', horizontalalignment='center')

plt.xlabel('date')
plt.ylabel('30-yr swap rate')
plt.title('original vs. constructed')
plt.legend(labels=['constructed 30-yr', 'original 30-yr'], loc='best')

plt.show()

# In[ ]:
