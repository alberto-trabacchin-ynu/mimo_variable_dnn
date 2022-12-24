import unittest
import deep_nn
import numpy as np
from pathlib import Path
import pandas as pd


class TestDeepNN(unittest.TestCase):

    @unittest.skip("Need to change data format")
    def test_load_iris_data_hold_out(self):
        file_path = Path("dataset/iris.data")
        test_size = 0.2
        random_state = 42
        x_train, y_train, x_test, y_test = deep_nn.load_iris_data_hold_out(file_path, test_size)
        print(x_train.shape)

    @unittest.skip("Need to change data format")
    def test_load_iris_data_kfold(self):
        file_path = Path("dataset/iris.data")
        X_train, y_train, X_test, y_test = deep_nn.load_iris_data_kfold(file_path, 5, True)
        
    @unittest.skip("Need to change data format")
    def test_init_parameters(self):
        m = 5
        n = 3
        p = 2
        layers_dim = [n, 3, p]
        params = deep_nn.init_parameters(layers_dim, m)

    # Then test also forward_prop
    @unittest.skip("Need to change data format")
    def test_full_forward_prop(self):
        layers_dim = [2, 7, 3]
        activations = ["sigmoid", "softmax"]
        m = 5
        params = deep_nn.init_parameters(layers_dim, m)
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        params["Z"], params["A"] = deep_nn.full_forward_prop(X, params, layers_dim, activations)

    @unittest.skip("Need to change data format")
    def test_compute_cost(self):
        Y = 0.8 * np.ones((2, 3))
        Y_hat = 0.3 * np.ones((2, 3))
        exp_cost = 2.3870992
        cost = deep_nn.compute_cost(Y_hat, Y)
        self.assertAlmostEqual(cost, exp_cost)

    @unittest.skip("Need to change data format")
    def test_Dsigmoid(self):
        params = {}
        params["Z"] = 0
        dAdZ = deep_nn.sigmoid_backward(params["Z"])

    @unittest.skip("Need to change data format")
    def test_softmax(self):
        Z = np.array([[0.7, 0.1, 0.4],
                      [0.5, 0.2, 0.9]])
        A = deep_nn.softmax(Z)

    #Define a function to init deterministic input parameters (X)
    @unittest.skip("Need to change data format")
    def test_full_back_prop(self):
        layers_dim = [2, 7, 3]
        activations = ["sigmoid", "softmax"]
        m = 5
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        params = deep_nn.init_parameters(layers_dim, m)
        grads = deep_nn.init_gradients(layers_dim, m)
        params["Z"], params["A"] = deep_nn.full_forward_prop(X, params, layers_dim, activations)
        # Then define specific function to calculate dA
        dA = 0.3 * np.ones((3, m))
        grads = deep_nn.full_back_prop(dA, params, grads, layers_dim, X)

    @unittest.skip("Need to change data format")
    def test_update_parameters(self):
        layers_dim = [2, 7, 3]
        m = 5
        params = deep_nn.init_parameters(layers_dim, m)
        grads = deep_nn.init_gradients(layers_dim, m)
        alpha = 0.1
        params = deep_nn.update_parameters(params, grads, alpha)

    @unittest.skip("Need to change data format")
    def test_train_model(self):
        layers_dim = [2, 15, 8, 4, 3]
        activations = ["sigmoid", "sigmoid", "sigmoid", "softmax"]
        m = 5
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        Y = np.array([[1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1]])
        params= deep_nn.init_parameters(layers_dim, m)
        params, losses = deep_nn.train_model(X, Y, params, layers_dim, activations, 
                                             alpha=0.1, n_iters=1e4, step_save=100, verbose=100)

    @unittest.skip("Need to change data format")
    def test_predict(self):
        layers_dim = [2, 5, 4, 3]
        activations = ["sigmoid", "sigmoid", "softmax"]
        m = 5
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        x = np.array([0.3, 0.3]).reshape((-1, 1))
        Y = np.array([[1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1]])
        params = deep_nn.init_parameters(layers_dim, m)
        params, losses = deep_nn.train_model(X, Y, params, layers_dim, activations, 
                                             alpha=0.1, n_iters=1e4, verbose=100)
        #Y_hat = deep_nn.predict(x, params, layers_dim, activations)
        #print(Y_hat)

    @unittest.skip("Need to change data format")
    def test_iris_prediction(self):
        X_train, Y_train, X_test, Y_test = deep_nn.load_iris_data(Path("dataset/iris.data"), 0.3)
        X_train = X_train
        X_test = X_test
        m = X_train.shape[1]
        n = X_train.shape[0]
        p = Y_train.shape[0]
        layers_dim = [n, 130, 8, p]
        activations = ["sigmoid", "sigmoid", "softmax"]
        params = deep_nn.init_parameters(layers_dim, m)
        params, losses = deep_nn.train_model(X_train, Y_train, params, layers_dim, activations,
                                             alpha=0.5, n_iters=1e5, verbose=1e3)
        test_cost = deep_nn.test_model(X_test, Y_test, params, layers_dim, activations)
        print(test_cost)

    @unittest.skip("Need to change data format")
    def test_plot_losses(self):
        step_save = 1
        iters = np.arange(0, 1e3)
        losses = np.exp(iters / 1e2)
        #print(losses)
        deep_nn.plot_losses(losses, step_save)

    @unittest.skip("Need to change data format")
    def test_iris_prediction(self):
        X_train, Y_train, X_test, Y_test = deep_nn.load_iris_data_hold_out(Path("dataset/iris.data"), 0.3)
        m = X_train.shape[1]
        n = X_train.shape[0]
        p = Y_train.shape[0]
        layers_dim = [n, 130, 8, p]
        activations = ["sigmoid", "sigmoid", "softmax"]
        step_save = 10
        n_iters = 1e4
        params = deep_nn.init_parameters(layers_dim, m)
        params, losses = deep_nn.train_model(X_train, Y_train, params, layers_dim, activations,
                                             alpha=0.1, n_iters=n_iters, verbose=step_save)
        deep_nn.plot_losses(losses, step_save)

    unittest.skip("Need to change data format")
    def test_cross_validate(self):
        X_train, Y_train, X_test, Y_test = deep_nn.load_iris_data_kfold(
                                           Path("dataset/iris.data"), n_splits=5, shuffle=True)
        layers_dim = [4, 130, 8, 3]
        activations = ["sigmoid", "sigmoid", "softmax"]
        step_save = 10
        n_iters = 1e3
        alpha = 0.2
        file_path = Path("dataset/iris.data")
        X_train, Y_train, X_test, Y_test = deep_nn.load_iris_data_kfold(file_path, 5, True)
        best_params, error = deep_nn.cross_validate(X_train, X_test, Y_train, Y_test, layers_dim, 
                                                     activations, alpha, n_iters, step_save, verbose=100)
        print("Best: " + str(error["losses"][-1]))

if __name__ == '__main__':
    unittest.main()