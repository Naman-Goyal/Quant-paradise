import datetime
import warnings

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import matplotlib.ticker

from objFunc1 import *
from objFunc1 import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Linear Regression
# 3D plot
# igore warning
warnings.filterwarnings("ignore")

df = pd.read_csv('swapLiborData.csv')

for i in range(df.shape[0]):
    df.loc[i, 'Date'] = pd.to_datetime(
        '1899-12-30') + pd.to_timedelta(df.loc[i, 'Date'], 'D')

lenT = len(df)

yY = df.iloc[:lenT, 8]
xX = df.iloc[:lenT, 6]

a = np.arange(-200, 200 + 5, 5.0)
len1 = len(a)
b = np.arange(-200, 200 + 5, 5.0)
len2 = len(b)
s = np.zeros((len2, len1))

a, b = np.meshgrid(a, b)

for i in range(len1):
    for j in range(len2):
        s[j, i] = np.sum((yY - a[j, i] - b[j, i] * xX) ** 2) / (2 * lenT)

# #surface
fig = plt.figure(figsize=(8, 4))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(
    a,
    b,
    s,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False)

ax.zaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10000))

plt.show()

# #contour

plt.figure(figsize=(8, 5))  # set the figure size
plt.contour(a, b, s, 40)

plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))

plt.show()
