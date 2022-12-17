import numpy as np


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0, Z)
    return A

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

