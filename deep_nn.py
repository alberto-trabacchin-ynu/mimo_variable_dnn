import numpy as np

def init_parameters(layers_dim):
    W = []
    b = []
    Z = []
    A = []
    for l in range(len(layers_dim) - 1):
        W.append(np.random.rand(layers_dim[l+1], layers_dim[l]))
        b.append(np.zeros((layers_dim[l+1], 1)))
    params = {
        "W": W,
        "b": b,
        "Z": Z,
        "A": A
    }
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
    return A, Z

def full_forward_prop(X, Y, params, layers_dim):
    A_prev = X
    for l in range(len(layers_dim)):
        A, Z = forward_prop(A_prev, params["W"][l], params["b"][l])
        params["A"][l] = A
        params["Z"][l] = Z
        A_prev = A
    return params

def back_prop(dA, W, Z, A_prev, m):
    dZ = np.multiply(dA, Dsigmoid(Z))
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

#Change indexing method (NOT CLEAR!)
def full_back_prop(dA, caches, params, layers_dim):
    grads = {}
    m = caches[0]["Z"].shape[1]
    for l in range(len(layers_dim), 1, -1):
        W = params["W" + str(l-1)]
        Z = caches[l-2]["Z"]
        A_prev = caches[l-2]["A"]
        dA_prev, dW, db = back_prop(dA, W, Z, A_prev, m)
        grads["dW" + str(l-1)] = dW
        grads["db" + str(l-1)] = db
        dA = dA_prev
    return grads

def update_params(params, grads, alpha):
    for i in range(len(params)):
        params["W" + str(i+1)] = params["W" + str(i+1)] - alpha * grads["dW" + str(i+1)]
        params["b" + str(i+1)] = params["b" + str(i+1)] - alpha * grads["db" + str(i+1)]
    return params

def compute_cost(Y, Y_hat):
    m = Y.shape[1]
    loss = np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat))
    loss = - np.sum(loss) / m
    return loss

def train_model(X, Y, params, layers_dim):
    Y_hat, caches = full_forward_prop(X, Y, params, layers_dim)
    #dA = - np.divide(Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
    #grads = full_back_prop(dA, caches, params, layers_dim)
    #params = update_params(params, grads, 0.1)
    
    return grads