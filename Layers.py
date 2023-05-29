import numpy as np

class Dense():

    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def feedforward(self, input):
        self.input = input
        return self.weights @ input + self.biases 

class Activation():

    def __init__(self, fn) -> None:
        self.fn = fn

    def feedforward(self, input):
        return self.fn(input)
