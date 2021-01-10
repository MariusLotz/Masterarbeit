from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from Interpolation.get_data import get_data
Learn_Data= get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')
typeY=3

Y = []
X1 = []
X2 = []

start = 0
for i in range(start, start + 25):
        X1.append(Learn_Data[0][i][1])
        X2.append(Learn_Data[0][i][2])
        Y.append(Learn_Data[1][i][typeY])
print(X1)

"""Plotting"""
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(X1, X2, Y, c=Y, cmap='seismic')
plt.show()
