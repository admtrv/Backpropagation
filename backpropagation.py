# backpropagation.py

import numpy as np
import config

# config data
weight_factor = config.weight_factor
learning_rate = config.learning_rate
momentum = config.momentum

# activation functions to add non-linearity to the neural net

def sigmoid(x):                                 # sigmoid activation function
    return 1 / (1 + np.exp(-x))

def tanh(x):                                    # hyperbolic tangens activation function
    return np.tanh(x)

def relu(x):                                    # rectified linear unit activation function
    return np.maximum(0, x)

def sigmoid_derivative(x):                      # derivations of functions above
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (x > 0).astype(float)

# loss function measures how much net's predictions deviate from true values

def mse_loss(y_true, y_pred):                   # mean squared error loss function
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# linear layer performs transformation of input data
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * weight_factor     # weights matrix
        self.bias = np.zeros((1, output_size))                                      # bias offset vector
        self.momentum = momentum                                                    # momentum factor, helps stabilize learning

        # from config
        self.learning_rate = learning_rate

        # previous updates
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)

    def forward(self, x):                                       # output = x * weights + bias
        self.input = x                                          # layer output calculation
        return np.dot(x, self.weights) + self.bias

    #  gradients used to adjust net parameters so errors reduced
    def backward(self, grad_output):                            # backward gradients
        grad_input = np.dot(grad_output, self.weights.T)        # calculation of gradients to adjust weights and offsets
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Update weights with momentum
        self.weight_momentum = self.momentum * self.weight_momentum - self.learning_rate * grad_weights
        self.bias_momentum = self.momentum * self.bias_momentum - self.learning_rate * grad_bias

        self.weights += self.weight_momentum
        self.bias += self.bias_momentum

        return grad_input

# activation layer adds non-linearity to the neuronal net
class ActivationLayer:
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        self.input = x
        return self.activation(x)

    def backward(self, gradient_output):
        return gradient_output * self.activation_derivative(self.input)