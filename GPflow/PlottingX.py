import matplotlib.pyplot as plt
from Interpolation.get_data import get_data
Learn_Data= get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')
typeX=0
typeY=3

Y = []
X = []
for i in range(len(Learn_Data[1])):
    if(i % 25 == 12):
        X.append(Learn_Data[0][i][typeX])
        Y.append(Learn_Data[1][i][typeY])
print(X)
#Y = [[Learn_Data[1][i][typeY]] for i in range(len(Learn_Data[1]))]
#X = [[Learn_Data[0][i][typeX]] for i in range(len(Learn_Data[0]))]

#fig, ax = plt.subplots()
#z = ax.plot(X,Y)
plt.scatter(X, Y)
plt.show()