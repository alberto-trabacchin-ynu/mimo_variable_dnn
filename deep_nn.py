import numpy as np


def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def feed_forward(W, b, A):
    return np.dot(W, A) + b

# Add possibility to change activation functions
def full_feed_forward(layers_dim, params, X):
    A_prev = X
    for i in range(1, len(layers_dim)):
        params["Z" + str(i)] = feed_forward(params["W" + str(i)], params["b" + str(i)], A_prev)
        params["A" + str(i)] = sigmoid(params["Z" + str(i)])
        A_prev = params["A" + str(i)]
    return params

def cost_function(Y_hat, Y):
    m = Y_hat.shape[1]
    Y_diff = np.power(Y_hat - Y, 2)
    Y_rse = np.sqrt(Y_diff)
    Y_mrse = 1 / m * np.sum(Y_rse)
    return Y_mrse

def sigmoid_backward(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

