import numpy as np

def mse(v, u):
    return np.mean((v-u)**2)