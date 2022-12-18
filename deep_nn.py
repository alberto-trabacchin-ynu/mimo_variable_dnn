import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0, Z)
    return A

def forward_prop(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
    cache = Z
    return A, cache

