import unittest
import deep_nn
import numpy as np

class TestDeepNN(unittest.TestCase):

    @unittest.skip("Need to change data format")
    def test_feed_forward(self):
        params = {}
        params["A"] = np.ones((10, 5), dtype=float)
        params["W"] = 0.2 * np.ones((8, 10), dtype=float)
        params["b"] = 0.2 * np.ones((8, 1), dtype=float)
        Z_exp = 2.2 * np.ones((8, 5), dtype=float)
        params["Z"] = deep_nn.feed_forward(params["W"], params["b"], params["A"])
        self.assertTrue(np.allclose(params["Z"], Z_exp))

    def test_full_feed_forward(self):
        layers_dim = [2, 7, 3]
        m = 5
        params = deep_nn.init_parameters(layers_dim, m)
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        Y_exp = np.array( [[0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168],
                           [0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168],
                           [0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168]] )
        params["Z"], params["A"] = deep_nn.full_forward_prop(X, params, layers_dim)

    @unittest.skip("Need to change data format")
    def test_cost_function(self):
        Y = 2 * np.ones((2, 3))
        Y_hat = np.zeros((2, 3))
        exp_cost = 4
        cost = deep_nn.cost_function(Y_hat, Y)
        self.assertEqual(cost, exp_cost)

    @unittest.skip("Need to change data format")
    def test_sigmoid_backward(self):
        params = {}
        params["Z"] = 0
        dAdZ = deep_nn.sigmoid_backward(params["Z"])

    @unittest.skip("Need to change data format")
    def test_full_feed_backward(self):
        A = np.ones((2, 2))
        Z = np.ones((2, 2))
        dA, db = deep_nn.full_feed_backward(A, Z)
        print(dA)
        print()
        print(db)

    @unittest.skip("Need to change data format")
    def test_init_parameters(self):
        layers_dim = [3, 3, 2]
        params = deep_nn.init_parameters(layers_dim)

    @unittest.skip("Need to change data format")
    def test_compute_loss(self):
        Y = np.ones((2, 2))
        Y_hat = 0.5 * np.ones((2, 2))
        cost = deep_nn.compute_cost(Y, Y_hat)

    @unittest.skip("Need to change data format")
    def test_full_forward_prop(self):
        m = 10
        n = 3
        p = 2
        layers_dim = [n, 4, p]
        X = np.ones((n, m))
        Y = np.ones((2, 2))
        params = deep_nn.init_parameters(layers_dim, m)
        Y_hat, caches = deep_nn.full_forward_prop(X, Y, params, layers_dim)

    @unittest.skip("Need to change data format")
    def test_train_model(self):
        m = 10
        n = 3
        p = 2
        layers_dim = [n, 4, p]
        X = np.ones((n, m))
        Y = np.ones((p, m))
        params = deep_nn.init_parameters(layers_dim)
        grads = deep_nn.train_model(X, Y, params, layers_dim)
        print(grads["dW2"].shape)

    def test_init_parameters(self):
        m = 5
        n = 3
        p = 2
        layers_dim = [n, 3, p]
        params = deep_nn.init_parameters(layers_dim, m)

if __name__ == '__main__':
    unittest.main()