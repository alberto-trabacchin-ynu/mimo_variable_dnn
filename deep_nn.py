import numpy as np


def sigmoid(Z):
    cache = Z
    A = 1 / (1 + np.exp(-Z))
    return A, cache

def relu(Z):
    cache = Z
    A = np.maximum(0, Z)
    return A, cache

