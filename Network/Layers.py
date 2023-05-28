import numpy as np

class Dense:

    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def feedforward(self, input):
        return self.weights @ input + self.biases 

class Sigmoid:

    def __init__(self) -> None:
        self.sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))

    def feedforward(self, input):
        return self.sigmoid(input)
    
class Relu:

    def __init__(self) -> None:
        self.relu = np.vectorize(lambda x: max(0, x))

    def feedforward(self, input):
        return self.relu(input)