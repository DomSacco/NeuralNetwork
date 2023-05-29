import numpy as np

def mse(v, u):
    return np.mean((v-u)**2)

def mse_prime(u, v):
    return 2*(u-v)