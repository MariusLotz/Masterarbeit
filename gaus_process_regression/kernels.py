""" Collection of Covariance kernels """
import numpy as np

def k1(a, b, w):
    """ exp multi feature kernel """
    w = np.array([w], dtype=np.float64)
    M = w.T.dot(w)
    z = a - b
    x = z.dot(M.dot(z.T))
    f = np.exp(-x)
    return f

def k1_dw(a,b,w, partial):
    i = partial
    z = np.abs(a - b)
    w = np.array([w], dtype=np.float64)
    M = w.T.dot(w)
    x = z.dot(M.dot(z.T))
    y = -2 * z[i] * w.dot(z.T)
    f = y * np.exp(-x)
    return f

def k2(a, b, w):
    z =  np.abs(a - b)
    sum = (w[0] * z[0]) **2 + (w[1] * z[1]) **2 +  (w[2] * z[2]) **2
    f = np.exp(-sum)
    return f

def k2_dw(a, b, w, partial):
    i = partial
    z = a - b
    sum = (w[0] * z[0]) **2 + (w[1] * z[1]) **2 +  (w[2] * z[2]) **2
    f = - np.exp(-sum) * 2 * (w[i] **2) * z[i]
    return f

def k3(a, b, w):
    z =  np.abs(a - b)
    sum = w[0] * z[0] **2 + w[1] * z[1] **2 +  w[2] * z[2] **2
    f = np.exp(-sum)
    return f

def k3_dw(a, b, w, partial):
    i = partial
    z = np.abs(a - b)
    sum = w[0] * z[0] **2 + w[1] * z[1] **2 + w[2] * z[2] **2
    f = - np.exp(-sum) * 2 * w[i] * z[i]
    return f

def k4(a, b, w):
    z = a - b
    if np.linalg.norm(z) == 0:
        return 1
    else:
        sum = (w[0] * z[0] **2 + w[1] * z[1] **2 + w[2] * z[2] **2) **0.1
        f = np.exp(-sum)
        return f

def k4_dw(a, b, w, partial):
    i = partial
    z = a - b
    if np.linalg.norm(z) == 0:
        return 1
    else:
        sum = (w[0] * z[0] **2 + w[1] * z[1] **2 + w[2] * z[2] **2) **0.1
        factor = -(w[i] * z[i]) / (5 * (w[0] * z[0] **2 + w[1] * z[1] **2 + w[2] * z[2] **2) **0.9)
        f = np.exp(-sum) * factor
        return f

def k5(a, b, w):
    z = a - b
    if np.linalg.norm(z) == 0:
        return 1
    else:
        sum = w[0] * z[0] + w[1] * z[1] + w[2] * z[2]
        f = np.exp(-sum)
        return f

def k5_dw(a, b, w, partial):
    i = partial
    z = a - b
    factor = - w[i]
    if np.linalg.norm(z) == 0:
        return factor 
    else:
        sum = w[0] * z[0] + w[1] * z[1] + w[2] * z[2]
        f = np.exp(-sum) * factor
        return f

def k6(a, b, w):
    z = a - b
    if np.linalg.norm(z) == 0:
        return 1
    else:
        sum = w[0] * np.abs(z[0]) + w[1] * np.abs(z[1]) + w[2] * np.abs(z[2])
        f = (1/1 + sum) **2
        return f

def k6_dw(a, b, w, partial):
    i = partial
    z = a - b
    if np.linalg.norm(z) == 0:
        return -2 * w[i]
    else:
        sum = w[0] * np.abs(z[0]) + w[1] * np.abs(z[1]) + w[2] * np.abs(z[2])
        f = -2 * (1/1 + sum) **3 * w[i]
        return f