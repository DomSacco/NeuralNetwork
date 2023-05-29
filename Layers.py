import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self) -> None:
        self.input = None

    @abstractmethod
    def feedforwrd(self, input):
        pass

    @abstractmethod
    def backpropagation(self, gradient_output, learning_rate):
        pass

class Dense(Layer):

    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def feedforward(self, input):
        self.input = input
        return self.weights @ input + self.biases 

class Activation(Layer):

    def __init__(self, fn) -> None:
        self.fn = fn

    def feedforward(self, input):
        return self.fn(input)
