import gpflow
import scipy.interpolate as pol
import numpy as np
import tensorflow as tf
from Interpolation.get_data import get_data

def main(type=1):
    """ Data """
    Learn_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt')
    Test_Data = get_data('/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt')

    # Add European Price as Input
    for i in range(len(Learn_Data[0])):
        Learn_Data[0][i].append(Learn_Data[1][i][0])
        if i < len(Test_Data[0]):
            Test_Data[0][i].append(Test_Data[1][i][0])

    X_learn = np.array(Learn_Data[0])


    fX_learn = np.array([[Learn_Data[1][i][type]] for i in range(len(Learn_Data[1]))])

    X_test= np.array(Test_Data[0])
    fX_test = np.array([[Test_Data[1][i][type]] for i in range(len(Test_Data[1]))])

    """Learning GPR"""
    kernel = gpflow.kernels.RBF()
    meanfunction = gpflow.mean_functions.Constant(0)
    m = gpflow.models.GPR((X_learn, fX_learn), gpflow.kernels.RBF(), meanfunction, 10 ** (-5.99))
    # gpflow.utilities.print_summary(m) # Model information
    opt = gpflow.optimizers.NaturalGradient  # optimizing parameters

    # GPR Approximation of Test_Data:
    fX_test_GPR = m.predict_f(X_test)[0]


    """Linear Interpolation"""
    polater = pol.LinearNDInterpolator(X_learn, fX_learn)

    # Linear Approximation of Test Prices:
    fX_test_pol = polater.__call__(X_test)
    #print(fX_test_pol.shape, fX_test_GPR.shape )


    """See Maximal Difference"""
    max = 0
    min = 0
    max_X = 0
    min_X = 0
    for i in range(len(fX_test)):
        res = fX_test_pol[i] - fX_test[i] # Polater_res
        # res = fX_test_GPR[i]- fX_test[i] # GPR_res
        if (res > max):
            max = res
            max_X = X_test[i]

        if (res < min):
            min = res
            min_X = X_test[i]


    tf.print(max, min)
    print(max_X, min_X)

if __name__=="__main__":
    main()

