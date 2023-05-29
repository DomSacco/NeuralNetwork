#import numpy as np
from Functions.Loss import mse, mse_prime

class Neuronet:

    def __init__(self, *layers, learning_rate = 0.01, cost_fn = mse, cost_fn_prime = mse_prime) -> None:
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def predict(self, x):
        for l in self.layers:
            x = l.feedforward(x)
        return x
    
    def update(self, x):
        for l in self.layers[::-1]:
            x = l.backpropagation(x, self.learning_rate)

    def train(self, x_train, y_train, epochs, batch_size, verbose=True):
        for e in range(epochs):
            error = count = batch_count = 0
            for x, y in zip(x_train, y_train):

                #forward
                y_pred = self.predict(x)

                count += 1

                #loss
                error += self.cost_fn(y_pred, y)

                if batch_size == count:
                    batch_count += 1
                    #print(f'Batch {batch_count} compleated')
                    self.update(mse_prime(y_pred, y))
                    count = 0
                
            error /= len(x_train)
            if verbose:
                print(f'Epoch {e+1} error: {error}')
    
    '''
    def test(self, x_test, y_test, verbose=True):
        accuracy = 0 
        for x, y in zip(x_test, y_test):
            y_pred = self.predict(x)
    '''

