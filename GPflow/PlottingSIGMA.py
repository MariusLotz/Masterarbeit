import matplotlib.pyplot as plt
from Interpolation.get_data import get_data
Learn_Data= get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')
typeX=1
typeY=3

Y = []
X = []
start = 100
for i in range(start, start + (len(Learn_Data[1])//5)):
    if(i % 5 == 0):
        X.append(Learn_Data[0][i][typeX])
        Y.append(Learn_Data[1][i][typeY])
print(X)

plt.scatter(X, Y)
plt.show()