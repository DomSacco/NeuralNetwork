import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self) -> None:
        self.input = None

    @abstractmethod
    def feedforward(self, input):
        pass

    @abstractmethod
    def backpropagation(self, gradient_output):
        pass

class Dense(Layer):

    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def feedforward(self, input):
        self.input = input
        return self.weights @ input + self.biases 
    
    def backpropagation(self, gradient_output, learning_rate):
        nabla_w = np.outer(gradient_output, self.input) # gradient_output.T @ input
        nabla_b = gradient_output   #
        self.weights -= learning_rate * nabla_w
        self.biases -= learning_rate * nabla_b
        gradient_input = self.weights.T @ gradient_output
        return gradient_input

class Activation(Layer):

    def __init__(self, fn, fn_prime) -> None:
        self.fn = fn
        self.fn_prime = fn_prime

    def feedforward(self, input):
        self.input = input
        return self.fn(input)
    
    def backpropagation(self, gradient_output, learning_rate):
        return gradient_output * self.fn_prime(self.input)
