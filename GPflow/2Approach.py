import gpflow
import numpy as np
import tensorflow as tf
from Interpolation.get_data import get_data
from Interpolation.linear_interpolation import interpolater

""" Data """
Learn_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt')
Test_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')

type = 3 # American Premium
X = np.array(Learn_Data[0]) # log(S/K=100), sigma, q
fX = np.array([[Learn_Data[1][i][type]] for i in range(len(Learn_Data[1]))])
X.reshape(len(X), len(X[0]))
fX.reshape(len(fX), 1)

X_test = np.array(Test_Data[0]) # log(S/K=100), sigma, q
fX_test = np.array([[Test_Data[1][i][type]] for i in range(len(Test_Data[1]))])
X_test.reshape(len(X_test), len(X_test[0]))
fX_test.reshape(len(fX_test), 1)


"""Defining Model"""
kernel = gpflow.kernels.RBF()
meanfunction = gpflow.mean_functions.Constant(1)
#likelihood = gpflow.likelihoods.Gaussian()
#m = gpflow.models.VGP((X, fX), kernel, likelihood)
m = gpflow.models.GPR((X, fX), gpflow.kernels.RBF(), meanfunction, 10**(-5))
opt = gpflow.optimizers.NaturalGradient # optimizing parameters
out = m.predict_f(X_test)

"""Polater Modell"""
data_for_polater = Learn_Data
real_data = Test_Data
x_list = real_data[0]
size = len(x_list)
y_list = [real_data[1][i][type] for i in range(size)]
polater = interpolater(type, data_for_polater[0], data_for_polater[1])
y1_list = []
# Creating Testprices:
for i in range(size):
    y1_list.append(polater.__call__(X_test[i]))

Y1 = np.array(y1_list)
Y1.reshape(len(Y1),1)


#tf.print(tf.math.subtract(out[0], fX_test))

min = 0
max = 0
for i in range(len(fX_test)):
    res = 0.5 * (out[0][i] + Y1[i]) - fX_test[i]
    if (res > max):
        max = res
        max_X = X_test[i]

    if (res < min):
        min = res
        min_X = X_test[i]

tf.print(max,min)
print(max_X, min_X)
