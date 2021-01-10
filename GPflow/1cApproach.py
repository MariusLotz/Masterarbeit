import gpflow
import numpy as np
import tensorflow as tf
from Interpolation.get_data import get_data

""" getting Data """
Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt')
X_list = Data[0]
for i in range(len(X_list)):
    X_list[i].append(Data[1][i][0]) # Add European Price as Input

X = np.array(X_list) # log(S/K=100), sigma, q
print(X)
type = 2
fX = np.array([[Data[1][i][type]] for i in range(len(Data[1]))]) # American Call Prices

X.reshape(len(X), len(X[0]))
fX.reshape(len(fX), 1)

"""Defining Model"""

mean_function = gpflow.mean_functions.Constant(1)
m = gpflow.models.GPR((X, fX), gpflow.kernels.RBF(), mean_function)
opt = gpflow.optimizers.Scipy() # optimizing parameters

out = m.predict_f(X)

tf.print(out[0])