import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)* (1 - sigmoid(x))

def relu(x):
    return np.fmax(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

