import numpy as np

#Then separate init_parameters with init_gradients
# Pass n_layers=L
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

# Change indexing method (NOT CLEAR!)
# Pass n_layers=L instead of layers_dim
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

# Do not pass grads for external. Define specific function to initialize grads to use internally.
def train_model(X, Y, params, grads, layers_dim, alpha, n_iters):
    L = len(layers_dim) - 2
    losses = []
    for iter in range(int(n_iters)):
        params["Z"], params["A"] = full_forward_prop(X, params, layers_dim)
        Y_hat = params["A"][L]
        losses.append(compute_cost(Y, Y_hat))
        dA = - np.divide(Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
        grads = full_back_prop(dA, params, grads, layers_dim, X)
        params = update_parameters(params, grads, alpha)
        #print(f"Iteration {iter + 1}: " + "[{0:.6f}]".format(losses[iter]))
    return params, losses