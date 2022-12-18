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

def full_forward_prop(X, Y, params, layers_dim):
    caches = []
    A_prev = X
    for l in range(len(layers_dim) - 1):
        A, cache = forward_prop(A_prev, params["W" + str(l+1)], params["b" + str(l+1)])
        A_prev = A
        caches.append(cache)
    return A, caches

def back_prop(dA, W, Z, A_prev, m):
    dZ = np.multiply(dA, Dsigmoid(Z))
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def compute_cost(Y, Y_hat):
    m = Y.shape[1]
    loss = np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat))
    loss = - np.sum(loss) / m
    return loss

def train_model(X, Y, params, layers_dim):
    Y, caches = full_forward_prop(X, Y, params, layers_dim)