import numpy as np
import pylab as pb
import GPy

def main():
    d = 1 # input dimension
    var = 1. # variance
    theta = 0.2 # lengthscale
    k = GPy.kern.RBF(d, var, theta)


    X = np.linspace(0., 1., 500)  # 500 points evenly spaced over [0,1]
    X = np.linspace(0.05, 0.95, 10)[:, None]
    Y = -np.cos(np.pi * X) + np.sin(4 * np.pi * X) + np.random.randn(10, 1) * 0.05

    print(type(X))
    m = GPy.models.GPRegression(X, Y, k)
    # print(m)

    # m.optimize_restarts(num_restarts=10)

    print(m.predict(np.array(0.5)))




if __name__=="__main__":
    main()