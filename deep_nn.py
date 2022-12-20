import numpy as np

#Then separate init_parameters with init_gradients
def init_parameters(layers_dim, m):
    params = {
        "W": [],
        "b": [],
        "Z": [],
        "A": []
    }
    grads = {
        "dW": [],
        "db": [],
    }
    for l in range(len(layers_dim) - 1):
        params["W"].append(2 * np.random.rand(layers_dim[l+1], layers_dim[l]) - 1)
        params["b"].append(np.zeros((layers_dim[l+1], 1)))
        params["Z"].append(np.zeros((layers_dim[l+1], m)))
        params["A"].append(np.zeros((layers_dim[l+1], m)))
        grads["dW"].append(np.zeros((layers_dim[l+1], layers_dim[l])))
        grads["db"].append(np.zeros((layers_dim[l+1], 1)))
    return params, grads

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

def full_forward_prop(X, params, layers_dim):
    A_prev = X
    for l in range(len(layers_dim) - 1):
        A, Z = forward_prop(A_prev, params["W"][l], params["b"][l])
        params["A"][l] = A
        params["Z"][l] = Z
        A_prev = A
    return params["Z"], params["A"]

def back_prop(dA, W, Z, A_prev, m):
    dZ = np.multiply(dA, Dsigmoid(Z))
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

#Change indexing method (NOT CLEAR!)
def full_back_prop(dA, params, grads, layers_dim):
    m = params["Z"][0].shape[1]
    for l in range(len(layers_dim) - 1, 0, -1):
        W = params["W"][l-1]
        Z = params["Z"][l-1]
        A_prev = params["A"][l-1]
        dA_prev, dW, db = back_prop(dA, W, Z, A_prev, m)
        grads["dW"][l-1] = dW
        grads["db"][l-1] = db
        dA = dA_prev
    return grads

def update_parameters(params, grads, alpha):
    for i in range(len(params)):
        params["W"] = [W - alpha * dW for W, dW in zip(params["W"], grads["dW"])]
        params["b"] = [b - alpha * db for b, db in zip(params["b"], grads["db"])]
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