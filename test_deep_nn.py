import unittest
import deep_nn
import numpy as np

class TestDeepNN(unittest.TestCase):

    def test_init_parameters(self):
        m = 5
        n = 3
        p = 2
        layers_dim = [n, 3, p]
        params = deep_nn.init_parameters(layers_dim, m)

    # Then test also forward_prop
    def test_full_forward_prop(self):
        layers_dim = [2, 7, 3]
        m = 5
        params = deep_nn.init_parameters(layers_dim, m)
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        params["Z"], params["A"] = deep_nn.full_forward_prop(X, params, layers_dim)

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

    #Define a function to init deterministic input parameters (X)
    def test_full_back_prop(self):
        layers_dim = [2, 7, 3]
        m = 5
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        params = deep_nn.init_parameters(layers_dim, m)
        grads = deep_nn.init_gradients(layers_dim, m)
        params["Z"], params["A"] = deep_nn.full_forward_prop(X, params, layers_dim)
        # Then define specific function to calculate dA
        dA = 0.3 * np.ones((3, m))
        grads = deep_nn.full_back_prop(dA, params, grads, layers_dim, X)

    def test_update_parameters(self):
        layers_dim = [2, 7, 3]
        m = 5
        params = deep_nn.init_parameters(layers_dim, m)
        grads = deep_nn.init_gradients(layers_dim, m)
        alpha = 0.1
        params = deep_nn.update_parameters(params, grads, alpha)

    def test_train_model(self):
        layers_dim = [2, 7, 3]
        m = 5
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        Y = np.ones((3, m))
        params= deep_nn.init_parameters(layers_dim, m)
        params, losses = deep_nn.train_model(X, Y, params, layers_dim, alpha=0.1, n_iters=1e4, verbose=True)



if __name__ == '__main__':
    unittest.main()