import numpy as np
import gradient_descent as gd
import gaus_process_regression.gauss_regr as gauss_regr


def main():

    ### Specify testfunction, start point and errorbar:
    x0 = gausst0
    test_f = price_16Nov_test_price
    test_gradient_f = price_16Nov_test_price_gradient
    error = 10 **-4
    ###
    print("starting in", x0)
    test= gd.gradient_descent_method(x0, test_f, test_gradient_f , errorbar=error, alphabar=0.1, betabar=0.7, max_iter = 999)
    print(test)


###################################Tests:

#### working for errobar >= 10**-7
def test_f1(x):
    f = (x ** 2) - 5
    return f

def test_gradient_f1(x):
    df = 2 * x
    return df

t1 = 5



### working for errobar >= 10**-7
def test_f2(x):
    f = x[0] **2 + x[1] **6 + x[2] **4
    return f

def test_gradient_f2(x):
    df = np.transpose(np.array([2 * x[0], 6 * x[1] **5, 4 * x[2] **3]))
    return df

t2 = np.array([77, 3, 5])



### working
def test_f3(x):
    f = x[0] **2 + x[1] **2 + x[2] **3
    return f

def test_gradient_f3(x):
    df = np.array([2 * x[0] **1, 2 * x[1] **1, 3 * x[2] **2])
    return df

t3 = np.array([7, 1, 1])
# for different starting points doesnt stop in local min (0,0,0) and leads to overflow


###########################################################################
# Optimization of the weights for GPR:
def price_16Nov_test_price(w, data = gauss_regr.get_data()):
    f = gauss_regr.log_likelihood(data[0], data[1], w)
    print(-f)
    return -f

def price_16Nov_test_price_gradient(w, data = gauss_regr.get_data()):
    d1 = gauss_regr.partial_log_likelihood(data[0], data[1], w, partial=1)
    d2 = gauss_regr.partial_log_likelihood(data[0], data[1], w, partial=2)
    d3 = gauss_regr.partial_log_likelihood(data[0], data[1], w, partial=3)
    df = np.array([-d1,-d2,-d3])
    return df

gausst0 = [1, 1.2, 1.3]


if __name__ == '__main__':
    main()
