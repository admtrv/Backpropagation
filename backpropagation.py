# backpropagation.py

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import config

# config data
dataset = config.dataset
x_train = config.datasets[dataset]['x_train']
y_train = config.datasets[dataset]['y_train']

learning_rate = config.learning_rate
epochs = config.epochs
activation = config.activation

use_momentum = config.use_momentum
momentum = config.momentum

support_layer = config.use_support_layer
support_layer_in = config.support_layer_in
support_layer_out = config.support_layer_out

output_dir = config.output_dir
os.makedirs(output_dir, exist_ok=True)

current_time = int(time.time())
np.random.seed(current_time)

# activation functions to add non-linearity to the neural net
def sigmoid(x):                     # sigmoid
    return 1 / (1 + np.exp(-x))

def tanh(x):                        # hyperbolic tangens
    return np.tanh(x)

def relu(x):                        # rectified linear unit
    return np.maximum(0, x)

# derivations of functions above
def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - x ** 2

def relu_derivative(x):
    return (x > 0).astype(float)

def get_activation_function(name):
    if name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif name == 'tanh':
        return tanh, tanh_derivative
    elif name == 'relu':
        return relu, relu_derivative

# loss function measures how much net's predictions deviate from true values
def mse_loss(y_true, y_pred):                   # mean squared error
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# linear layer performs transformation of input data
class LinearLayer:
    def __init__(self, input_size, output_size, activation='tahn', use_momentum=False):
        # weights
        if activation == 'sigmoid' or activation == 'tanh':
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        elif activation == 'relu':
            std_dev = np.sqrt(2 / input_size)
            self.weights = np.random.randn(input_size, output_size) * std_dev

        # bias offsets
        self.bias = np.zeros((1, output_size))
        self.learning_rate = learning_rate

        # momentum
        self.momentum = momentum                                # momentum factor, helps stabilize learning
        self.use_momentum = use_momentum

        # previous updates
        if self.use_momentum:
            self.weight_momentum = np.zeros_like(self.weights)
            self.bias_momentum = np.zeros_like(self.bias)

    def forward(self, x):                                       # layer output calculation
        self.input = x
        return np.dot(x, self.weights) + self.bias              # output = x * weights + bias

    #  gradients used to adjust net parameters so errors reduced
    def backward(self, gradient_output):                            # calculation of gradients to adjust weights and offsets
        gradient_input = np.dot(gradient_output, self.weights.T)
        gradient_weights = np.dot(self.input.T, gradient_output)
        gradient_bias = np.sum(gradient_output, axis=0, keepdims=True)

        # update weights with momentum
        if self.use_momentum:
            self.weight_momentum = self.momentum * self.weight_momentum - self.learning_rate * gradient_weights
            self.bias_momentum = self.momentum * self.bias_momentum - self.learning_rate * gradient_bias
            self.weights += self.weight_momentum
            self.bias += self.bias_momentum
        else:
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

        return gradient_input

# activation layer adds non-linearity to the neuronal net
class ActivationLayer:
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward(self, gradient_output):
        return gradient_output * self.activation_derivative(self.output)

# neural net
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)


def train(model, epochs):
    loss_history = []

    for epoch in range(epochs):
        # forward pass
        predictions = model.forward(x_train)
        loss = mse_loss(y_train, predictions)
        loss_history.append(loss)

        # backward pass
        gradient_loss = mse_loss_derivative(y_train, predictions)
        model.backward(gradient_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    return loss_history

def evaluate(model):
    predictions = model.forward(x_train)
    rounded_predictions = np.round(predictions, decimals=4).astype(float)
    binary_predictions = np.round(predictions).astype(int)

    print("\nPredictions", rounded_predictions.flatten().tolist())
    print("Rounded", binary_predictions.flatten().tolist())
    print("Actual", y_train.flatten().tolist())

# plot metrics
def plot_metrics(loss_history, title):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    filename = re.sub(r'[^a-zA-Z0-9_]', '', re.sub(r'\s+', '_', title)).lower() + ".png"
    filename = re.sub(r'_+', '_', filename)

    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

if __name__ == '__main__':

    # get activation functions
    activation_func, activation_derivative = get_activation_function(activation)

    # get output activation function
    if activation == 'relu':
        output_activation_func, output_activation_derivative = sigmoid, sigmoid_derivative
        output_activation_name = 'sigmoid'
    else:
        output_activation_func, output_activation_derivative = activation_func, activation_derivative
        output_activation_name = activation

    print(f"Training with...")
    print(f"  Dataset {dataset.upper()}")
    print(f"  Activation Function {activation.capitalize()}")
    print(f"  Output Activation Function {output_activation_name.capitalize()}")
    print(f"  Momentum {'Yes' if use_momentum else 'No'}")
    print(f"  Support Layer {'Yes' if support_layer else 'No'}")
    print(f"  Learning Rate {learning_rate}")
    print(f"  Epochs {epochs}\n")

    # get layers
    if not support_layer:
        layers = [
            LinearLayer(2, 4, activation=activation, use_momentum=use_momentum),    # input layer
            ActivationLayer(activation_func, activation_derivative),

            LinearLayer(4, 1, activation=activation, use_momentum=use_momentum),    # output layer
            ActivationLayer(output_activation_func, output_activation_derivative)
        ]
    else:
        layers = [
            LinearLayer(2, support_layer_in, activation=activation, use_momentum=use_momentum),         # input layer
            ActivationLayer(activation_func, activation_derivative),

            LinearLayer(support_layer_in, support_layer_out, activation=activation, use_momentum=use_momentum),  # support layer
            ActivationLayer(activation_func, activation_derivative),

            LinearLayer(support_layer_out, 1, activation=activation, use_momentum=use_momentum),       # output layer
            ActivationLayer(output_activation_func, output_activation_derivative)
        ]

    model = NeuralNet(layers)                                   # model
    loss_history = train(model, epochs)                         # train
    evaluate(model)                                             # forward

    title = f"{activation.capitalize()} Training Loss + Momentum {'Yes' if use_momentum else 'No'} + Support Layer {'Yes' if support_layer else 'No'} + Learning Rate {learning_rate} + Dataset {dataset.upper()}"
    plot_metrics(loss_history, title)                           # plot
