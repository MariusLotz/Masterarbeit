import gpflow
import scipy.interpolate as pol
import numpy as np
import tensorflow as tf
from Interpolation.get_data import get_data


def main(type=3):
    """ Data """
    Learn_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt')
    Test_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')

    X_learn = np.array(Learn_Data[0])

    fX_learn = np.array([[Learn_Data[1][i][type]] for i in range(len(Learn_Data[1]))])

    X_test = np.array(Test_Data[0])
    fX_test = np.array([[Test_Data[1][i][type]] for i in range(len(Test_Data[1]))])

    """Defining Kernel"""
    kernel2 = gpflow.kernels.Matern32() * gpflow.kernels.Linear()
    kernel2x = kernel2 + kernel2 + kernel2
    kernel4 = gpflow.kernels.RBF(active_dims=[0]) * gpflow.kernels.RBF(active_dims=[1]) * gpflow.kernels.RBF(active_dims=[2])
    kernelplus = gpflow.kernels.Exponential(active_dims=[0]) * gpflow.kernels.Exponential(active_dims=[1]) * gpflow.kernels.Exponential(active_dims=[2])
    kernel11 = gpflow.kernels.Matern32(active_dims=[0]) * gpflow.kernels.Matern32(active_dims=[1]) * gpflow.kernels.Matern32(active_dims=[2])
    kernel12 = gpflow.kernels.Linear() * (kernel11 + gpflow.kernels.Matern32(active_dims=[1]) * gpflow.kernels.Matern32(active_dims=[2]))
    kernel13 = kernel12 + kernel12 + kernel12 + kernel12
    kernel5 = kernel11 * kernel13
    kernel51 = (kernel11 + kernel11 + kernel11) * (kernel13 + gpflow.kernels.Linear())
    kernel52 = (kernel11 + kernel11 + kernel11) * kernel13

    k1 = gpflow.kernels.Matern32(active_dims=[0]) * gpflow.kernels.Matern32(active_dims=[1]) * gpflow.kernels.Matern32(active_dims=[2])
    k2 = gpflow.kernels.Linear() * (k1 + gpflow.kernels.Matern32(active_dims=[1]) * gpflow.kernels.Matern32(active_dims=[2]))

    F = (k1 + k1 + k1 + k1) * (k2 + k2 + k2 + k2 + k2)




    """Learning GPR"""
    meanfunction = gpflow.mean_functions.Constant(0)
    m = gpflow.models.GPR((X_learn, fX_learn), F, meanfunction, 10 ** (-5.99))
    gpflow.utilities.set_trainable(m.mean_function.c, False)
    gpflow.utilities.print_summary(m) # Model information
    opt = gpflow.optimizers.scipy  # optimizing parameters

    # GPR Approximation of Test_Data:
    fX_test_GPR = m.predict_f(X_test)[0]

    """Linear Interpolation"""
    polater = pol.LinearNDInterpolator(X_learn, fX_learn)

    # Linear Approximation of Test Prices:
    fX_test_pol = polater.__call__(X_test)
    # print(fX_test_pol.shape, fX_test_GPR.shape )

    """See Maximal Difference"""
    max = 0
    min = 0
    minS = 0
    max_I = 0
    min_I = 0
    min_IS = 0
    for i in range(len(fX_test)):
        # res = fX_test_pol[i] - fX_test[i]  # Polater_res
        res = fX_test_GPR[i]- fX_test[i] # GPR_res
        if (res > max):
            max = res
            max_I = fX_test[i], fX_test_GPR[i], X_test[i]

        if (res < min and fX_test_GPR[i]>0):
            min = res
            min_I = fX_test[i], fX_test_GPR[i], X_test[i]

        if (res < minS):
            minS = res
            min_IS = fX_test[i], fX_test_GPR[i], X_test[i]


    tf.print(max, max_I)
    tf.print(min, min_I)
    tf.print(minS, min_IS)



if __name__ == "__main__":
    main()