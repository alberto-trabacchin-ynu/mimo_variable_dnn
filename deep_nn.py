from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

def load_iris_data(ds_path, test_size):
    ds_path = Path(ds_path)
    if not ds_path.is_file():
        Path("dataset").mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        urllib.request.urlretrieve(url, ds_path)
    iris = pd.read_csv(ds_path).to_numpy()
    X = iris[:, 0:4]
    labels = iris[:, 4].reshape(-1, 1)
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)
    one_hot_enc = OneHotEncoder(handle_unknown="ignore")
    enc_labels = one_hot_enc.fit_transform(labels).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, enc_labels,
                                                        test_size=test_size, random_state=42)
    x_train = x_train.T.astype(float) / 10
    x_test = x_test.T.astype(float) / 10
    y_train = y_train.T.astype(float)
    y_test = y_test.T.astype(float)
    return x_train, y_train, x_test, y_test

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

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def Dsigmoid(Z):
    return np.multiply(sigmoid(Z), 1 - sigmoid(Z))

def relu(Z):
    A = np.maximum(0, Z)
    return A

def forward_prop(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "softmax":
        A = softmax(Z)
    return A, Z

def full_forward_prop(X, params, layers_dim, activations=None):
    A_prev = X
    L = len(layers_dim) - 2
    for l in range(L):
        params["A"][l], params["Z"][l] = forward_prop(A_prev, params["W"][l], params["b"][l], activations[l])
        A_prev = params["A"][l]
    params["A"][L], params["Z"][L] = forward_prop(A_prev, params["W"][L], params["b"][L], activations[L])
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

def train_model(X, Y, params, layers_dim, activations, alpha, n_iters, verbose=None):
    L = len(layers_dim) - 2
    m = X.shape[1]
    losses = []
    grads = init_gradients(layers_dim, m)
    for iter in range(1, int(n_iters) + 1):
        params["Z"], params["A"] = full_forward_prop(X, params, layers_dim, activations)
        Y_hat = params["A"][L]
        #dA = - np.divide(Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
        dA = Y_hat - Y
        grads = full_back_prop(dA, params, grads, layers_dim, X)
        params = update_parameters(params, grads, alpha)
        if verbose is not None and (iter % verbose == 0):
            losses.append(compute_cost(Y, Y_hat))
            print(f"Iteration {iter}: " + "[{0:.8f}]".format(losses[-1]))            
    return params, losses

def predict(x, params, layers_dim, activations):
    L = len(layers_dim) - 2
    params["Z"], params["A"] = full_forward_prop(x, params, layers_dim, activations)
    return params["A"][L]

def test_model(X_test, Y_test, params, layers_dim, activations):
    L = len(layers_dim) - 2
    params["Z"], params["A"] = full_forward_prop(X_test, params, layers_dim, activations)
    Y_hat = params["A"][L]
    cost = compute_cost(Y_test, Y_hat)
    return cost

def plot_losses(losses, step_save):
    iters = np.arange(0, len(losses) * step_save, step_save)
    plt.plot(iters, losses)
    plt.show()