#import numpy as np
from Network.Functions.Loss import mse

class Neuronet:

    def __init__(self, *layers, learning_rate = 0.01, cost_fn = mse) -> None:
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def predict(self, x):
        for l in self.layers:
            x = l.feedforward(x)
        return x

    def train(self, x_train, y_train, epochs, batch_size, verbose=True):
        for e in range(epochs):
            error = count = batch_count = 0
            for x, y in zip(x_train, y_train):

                #forward
                y_pred = self.predict(x)

                count += 1

                #loss
                error += self.cost_fn(y_pred, y)
                
            error /= len(x_train)
            if verbose:
                print(f'Epoch {e+1} error: {error}')
    
    '''
    def test(self, x_test, y_test, verbose=True):
        accuracy = 0 
        for x, y in zip(x_test, y_test):
            y_pred = self.predict(x)
    '''

