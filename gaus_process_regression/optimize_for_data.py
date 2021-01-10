import numpy as np
from gradient_descent.gradient_descent import gradient_descent_method as gd
import gaus_process_regression.gauss_regr as gauss
import gaus_process_regression.kernels as kern
"""Not working yet, test_gradient_descent for optimization"""


def optimize(w0=[0.1,0.1,0.1], data=gauss.get_data(), alphabar=0.1, betabar=0.7, error_gradient=0.1,
             cov_f=kern.k1, cov_df=kern.k1_dw, max_iter=999):

    # Trainings Data
    x, f = data
    x = np.array(x, dtype=np.float64)
    f = np.array(f, dtype=np.float64)


    # Minimize function:
    def func(w):
        func = log_likelihood(x, f, w, cov_f)
        return -f

    # Gradient:
    def gradient_f(w):
        n = len(w)
        grad =  np.zeros(n)
        for i in range(n):
            grad[i] = partial_log_likelihood(x, f, w, cov_f, cov_df, i)
        return -grad

    # Gradient descent method
    weights = gd(w0, func, gradient_f, error_gradient, alphabar, betabar,
                                      max_iter)

    return weights



def log_likelihood(x, f, w, cov_f):
    K = gauss.cov_matrix(x, w, cov_f)

    # Eigenvalue composition, d vector of eigenvalues and S matrix with eigenvectors
    d, S = np.linalg.eigh(K)
    n = len(f)

    # Parts of the formula:
    A = - 0.5 * f.dot(K_inv_f(f, d, S, n))
    B = - 0.5 * logdet_K(d, S, n)
    C = - (n/2) * np.log(2 * np.pi)

    out = A + B + C
    return out

def partial_log_likelihood(x, f, w, cov_f, cov_df, partial):
    K = np.array(gauss.cov_matrix(x, w, cov_f), dtype=np.float64)
    dK = np.array(gauss.cov_matrix(x, w, cov_f, cov_df, partial), dtype=np.float64)
    # Eigenvalue composition, d vector of eigenvalues and S matrix with eigenvectors
    d, S = np.linalg.eigh(K)
    n = len(f)
    print(K_inv_f(f, d, S, n))
    out = (K_inv_f(f, d, S, n).dot(K_inv_f(f, d, S, n).T) - K_inv(d, S, n)) * dK
    out = 0.5 * np.trace(out, dtype=np.float64)
    print(out)

    return out


def K_inv(d, S, n):
    out = np.zeros((n,n))
    for i in range(n):
        out[i,:] = (1/d[i]) * S[i,:]
    out = S.dot(out)
    return out


def K_inv_f(f, d, S, n):
    """Calculating K^-1 * f with K^-1 = S D^-1 S"""

    # out = S * f:
    out = S.dot(f.T)
    # out = D^-1 S * f:
    for i in range(n):
        if d[i] > 10 **-50:
            out[i] = (1/d[i]) * out[i]
        else:
            out[i] = 10 **-50 * out[i]
    # out = S D^-1 S * f:
    S.dot(out)
    return out

def logdet_K(d, S, n):
    """Calculating log(det(K)) = Tr(S ln(D) S)"""
    out = 0
    for i in range(n) :
        for j in range(n):
            if d[j] > 10 **-50:
                x = S[i][j] **2 * np.log(d[j])
            else:
                x = S[i][j] ** 2 * np.log(10 **-50)
            out += x
    return out


if __name__ =="__main__":
    optimize()
