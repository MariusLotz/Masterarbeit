"""Here are all functions connected to the gaus-process-regression except kernel functions"""
import numpy as np
import gaus_process_regression.kernels as kern


def cov_matrix(data_points, w, cov_f=kern.k5, cov_df=kern.k5_dw, partial=-1):
    """Creating initial covariance matrix and its derivative"""

    # Typecheck:
    if type(data_points) == list or type(data_points) == np.ndarray and type(partial) == int:

        # Creating zero matrix:
        size = (len(data_points))
        K = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                a = data_points[i]
                b = data_points[j]
                if partial < 0:
                    # Filling matrix with covariances depending on kernel:
                    K[i, j] = cov_f(a, b, w)
                # Filling d/dw_partial matrix:
                if partial >= 0:
                    K[i, j] = cov_df(a, b, w, partial)
        return K

    else: raise TypeError


def cov_matrix_y(data_points, y, w, cov_f=kern.k5):
    """ Covariance vector for one single (new) datapoint y"""

    # Typecheck:
    if type(data_points) == list or type(data_points) == np.ndarray:

        # Creating zero vector:
        size = len(data_points)
        K = np.zeros(size)

        # Filling covariance vector depending on kernel:
        for i in range(size):
            K[i] = cov_f(a, y, w)
        return K

    else: raise TypeError


def log_likelihood(x,fx,w):
    """ log(p(f(x)|x,w) = -0.5 fx^T K^-1 fx - 0.5 log(det(K)) -n/2 log(2pi)"""

    # Checking for right datatype:
    if type(fx) != np.ndarray or type(x) != np.ndarray or type(w) != np.ndarray:
        fx = np.array(fx)
        x = np.array(x)
        w = np.array(w)

    # log(p(f(x)|x,w):
    A = 0.5 * fx.dot(np.linalg.inv(cov_matrix(x,w)).dot(fx.T))
    B = 0.5 * np.linalg.slogdet(cov_matrix(x,w))[1] * np.linalg.slogdet(cov_matrix(x,w))[0]  #PROBLEM
    C = (len(fx)/2) * np.log(2 * np.pi)
    R = -(A + B + C)
    return R


def partial_log_likelihood(x, fx, w, partial=1):
    """partial log likelihood for given data"""

    # Checking for right datatype:
    if type(fx) != np.ndarray or type(x) != np.ndarray or type(w) != np.ndarray:
        fx = np.array(fx)
        x = np.array(x)
        w = np.array(w)

        # d/dpartial log(p(f(x)|x,w):
        K_inv = np.linalg.inv(cov_matrix(x,w))
        A = fx.dot(K_inv.dot(cov_matrix(x, w, partial).dot(K_inv.dot(fx.T))))
        B = -np.trace(K_inv.dot(cov_matrix(x, w, partial)))
        return (0.5 * (A+B))


def get_data(dest_file='/home/user/PycharmProjects/pythonProject/option_pricing/prices/price_E0E6p_16Nov.txt'):
    """Getting data out of a text file and convert to right data type"""

    file = open(dest_file, 'r')
    x_list = []
    fx_list = []

    while True:
        # Get next line from file
        line = file.readline()
        if not line:
            break

        # Training data x = (s,sigma,q) and fx = premium
        x = [np.log(float(line.strip('][').split(', ')[0])/100), # log(S/K=100)
             float(line.strip('][').split(', ')[2]), # sigma
             float(line.strip('][').split(', ')[3])] # q
        x_list.append(x)
        fx = float(line.strip('][').split(', ')[-2][:-2]) # alpha
        fx_list.append(fx)
    file.close()

    return [x_list, fx_list]


def regression_function(y, data, w, Kxx_inv):
    """Posteriori Regression function given trainings data and right weights w"""

    # Trainings data:
    x_list = data[0]
    fx_list = np.array(data[1])

    y = np.array(y)
    Kxy = cov_matrix_y(x_list, y, w)

    # R(y):= K(X,y) * K(X,X)^-1 * f(X) ~ f(y)
    f = Kxy.dot(Kxx_inv.dot(fx_list.T))
    return f








