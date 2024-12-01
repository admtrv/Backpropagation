# config.py

import numpy as np

# dataset config
dataset = 'xor'     # options: 'xor', 'or', 'and'
datasets = {
    'xor': {
        'x_train': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y_train': np.array([[0], [1], [1], [0]])
    },
    'or': {
        'x_train': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y_train': np.array([[0], [1], [1], [1]])
    },
    'and': {
        'x_train': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y_train': np.array([[0], [0], [0], [1]])
    }
}

# neural net config
activation = 'tanh'     # options: 'sigmoid', 'tanh', 'relu'

use_support_layer = True
support_layer_in = 4
support_layer_out = 4

use_momentum = True
momentum = 0.9

# train config
learning_rate = 0.1
epochs = 600

# directory config
output_dir = "temp"
