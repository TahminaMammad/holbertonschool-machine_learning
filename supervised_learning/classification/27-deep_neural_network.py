#!/usr/bin/env python3
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network performing multiclass classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(
            isinstance(l, int) and l > 0 for l in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, nodes in enumerate(layers):
            key_W = f"W{i + 1}"
            key_b = f"b{i + 1}"
            prev_nodes = nx if i == 0 else layers[i - 1]
            self.__weights[key_W] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights[key_b] = np.zeros((nodes, 1))

    @property
    def weights(self):
        return self.__weights

    @property
    def cache(self):
        return self.__cache

    def forward_prop(self, X):
        """Forward propagation with softmax for last layer"""
        self.__cache["A0"] = X
        for i in range(self.L):
            W = self.__weights[f"W{i + 1}"]
            b = self.__weights[f"b{i + 1}"]
            A_prev = self.__cache[f"A{i}"]
            Z = np.dot(W, A_prev) + b
            if i == self.L - 1:
                # Softmax activation for multiclass
                expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = expZ / np.sum(expZ, axis=0, keepdims=True)
            else:
                # Sigmoid activation for hidden layers
                A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{i + 1}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Compute cost using cross-entropy for multiclass"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A + 1e-8))  # add epsilon for stability
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent with softmax for output layer"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dA_prev = 0

        for i in reversed(range(self.L)):
            A_curr = cache[f"A{i + 1}"]
            A_prev = cache[f"A{i}"]
            W_curr = self.__weights[f"W{i + 1}"]

            if i == self.L - 1:
                dZ = A_curr - Y
            else:
                dZ = dA_prev * A_curr * (1 - A_curr)

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 0:
                W_next = weights_copy[f"W{i + 1}"]
                dA_prev = np.dot(W_next.T, dZ)

            self.__weights[f"W{i + 1}"] = W_curr - alpha * dW
            self.__weights[f"b{i + 1}"] = self.__weights[f"b{i + 1}"] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
            if verbose and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")
            if graph and (i % step == 0 or i == iterations):
                costs.append(self.cost(Y, A))
                iters.append(i)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
