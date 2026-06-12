import numpy as np

def sigmoid_binary(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1

def tanh(x):
    return np.tanh(x)

# Derivadas (para backpropagation)
def d_sigmoid_binary(x):
    return sigmoid_binary(x) * (1 - sigmoid_binary(x))

def d_sigmoid_bipolar(x):
    return 0.5 * (1 + sigmoid_bipolar(x)) * (1 - sigmoid_bipolar(x))

def d_tanh(x):
    return 1 - np.tanh(x)**2