import numpy as np

def init_parameters(layers_dim):
    params = {}
    for l in range(len(layers_dim) - 1):
        params["W" + str(l+1)] = np.random.rand(layers_dim[l+1], layers_dim[l])
        params["b" + str(l+1)] = np.zeros((layers_dim[l+1], 1))
    return params

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def Dsigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    A = np.maximum(0, Z)
    return A

def forward_prop(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
    cache = Z
    return A, cache

def back_prop(dA, W, Z, A_prev, m):
    dZ = np.multiply(dA, Dsigmoid(Z))
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

