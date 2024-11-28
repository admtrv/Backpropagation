# backpropagation.py

import numpy as np

def sigmoid(x):                                 # sigmoid function
    return 1 / (1 + np.exp(-x))

def tanh(x):                                    # hyperbolic tangens function
    return np.tanh(x)

def relu(x):                                    # rectified linear unit function
    return np.maximum(0, x)

def mse_loss(y_true, y_pred):                   # mean squared error function
    return np.mean((y_true - y_pred) ** 2)

def sigmoid_derivative(x):                      # derivations of functions above
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size