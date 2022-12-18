import unittest
import deep_nn
import numpy as np

class TestDeepNN(unittest.TestCase):

    def test_feed_forward(self):
        params = {}
        params["A"] = np.ones((10, 5), dtype=float)
        params["W"] = 0.2 * np.ones((8, 10), dtype=float)
        params["b"] = 0.2 * np.ones((8, 1), dtype=float)
        Z_exp = 2.2 * np.ones((8, 5), dtype=float)
        params["Z"] = deep_nn.feed_forward(params["W"], params["b"], params["A"])
        self.assertTrue(np.allclose(params["Z"], Z_exp))

    def test_full_feed_forward(self):
        layers_dim = [2, 5, 3]
        params = {}
        params["W1"] = 0.1 * np.ones((5, 2))
        params["b1"] = 0.1 * np.ones((5, 1))
        params["W2"] = 0.1 * np.ones((3, 5))
        params["b2"] = 0.1 * np.ones((3, 1))
        X = np.array([[0.3, 0.4, 0.1, -0.9, -0.2],
                      [-0.5, 0.1, 0.2, -0.6, 0.1]])
        Y_exp = np.array( [[0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168],
                           [0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168],
                           [0.56463498, 0.56677739, 0.56616644, 0.56063789, 0.56494168]] )
        params = deep_nn.full_feed_forward(layers_dim, params, X)
        layers_dim = [2, 5, 4]
        params = {}
        params["W1"] = 0.1 * np.ones((5, 2))
        params["b1"] = 0.1 * np.ones((5, 1))
        params["W2"] = 0.1 * np.ones((4, 5))
        params["b2"] = 0.1 * np.ones((4, 1))
        params = deep_nn.full_feed_forward(layers_dim, params, X)
        self.assertEqual(params["A2"].shape[0], 4)
        self.assertEqual(params["A2"].shape[1], 5)

    def test_cost_function(self):
        Y = 2 * np.ones((2, 3))
        Y_hat = np.zeros((2, 3))
        exp_cost = 4
        cost = deep_nn.cost_function(Y_hat, Y)
        self.assertEqual(cost, exp_cost)

    def test_sigmoid_backward(self):
        params = {}
        params["Z"] = 0
        dAdZ = deep_nn.sigmoid_backward(params["Z"])
        print(dAdZ)

if __name__ == '__main__':
    unittest.main()