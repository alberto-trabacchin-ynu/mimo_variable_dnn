import numpy as np

def init_parameters(layers_dim, m):
    params = {
        "W": [],
        "b": [],
        "Z": [],
        "A": []
    }
    for l in range(len(layers_dim) - 1):
        params["W"].append(2 * np.random.rand(layers_dim[l+1], layers_dim[l]) - 1)
        params["b"].append(np.zeros((layers_dim[l+1], 1)))
        params["Z"].append(np.zeros((layers_dim[l+1], m)))
        params["A"].append(np.zeros((layers_dim[l+1], m)))
    return params

def init_gradients(layers_dim, m):
    grads = {
        "dW": [],
        "db": [],
    }
    for l in range(len(layers_dim) - 1):
        grads["dW"].append(np.zeros((layers_dim[l+1], layers_dim[l])))
        grads["db"].append(np.zeros((layers_dim[l+1], 1)))
    return grads

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def Dsigmoid(Z):
    return np.multiply(sigmoid(Z), 1 - sigmoid(Z))

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

def full_back_prop(dA, params, grads, layers_dim, X):
    m = params["Z"][0].shape[1]
    params["A"].insert(0, X)
    for l in range(len(layers_dim) - 2, -1, -1):
        W = params["W"][l]
        Z = params["Z"][l]
        A_prev = params["A"][l]
        dA_prev, dW, db = back_prop(dA, W, Z, A_prev, m)
        grads["dW"][l] = dW
        grads["db"][l] = db
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

def train_model(X, Y, params, layers_dim, alpha, n_iters, verbose=None):
    L = len(layers_dim) - 2
    m = X.shape[1]
    losses = []
    grads = init_gradients(layers_dim, m)
    for iter in range(int(n_iters)):
        params["Z"], params["A"] = full_forward_prop(X, params, layers_dim)
        Y_hat = params["A"][L]
        dA = - np.divide(Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
        grads = full_back_prop(dA, params, grads, layers_dim, X)
        params = update_parameters(params, grads, alpha)
        if verbose is not None and (iter % verbose == 0):
            losses.append(compute_cost(Y, Y_hat))
            print(f"Iteration {iter}: " + "[{0:.8f}]".format(losses[-1]))            
    return params, losses