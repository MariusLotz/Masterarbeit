import gpflow
import numpy as np
import tensorflow as tf
from Interpolation.get_data import get_data

""" getting Data """
Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt')
X = np.array(Data[0]) # log(S/K=100), sigma, q
fX = np.array([[Data[1][i][1]] for i in range(len(Data[1]))]) # American Call Prices

X.reshape(len(X), len(X[0]))
fX.reshape(len(fX), 1)

"""Defining Model"""

m = gpflow.models.GPR((X, fX), gpflow.kernels.RBF())
opt = gpflow.optimizers.Scipy() # optimizing parameters

out = m.predict_f(X)

tf.print(out[0])
