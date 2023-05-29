import numpy as np

def mse(y_pred, y_act):
    return np.mean((y_pred-y_act)**2)

def mse_prime(y_pred, y_act):
    return 2*(y_pred-y_act)

def bcel(y_pred, y_act):
    return np.mean(-y_act * np.log(y_pred) - (1 - y_act) * np.log(1 - y_pred))

def bcel_prime(y_pred, y_act):
    return ((1 - y_act) / (1 - y_pred) - y_act / y_pred)