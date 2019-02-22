import numpy as np


class Model:
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.layers = []

    def add_layer(self, nodes):
        biases = np.random.random((nodes, 1))
        if len(self.layers) == 0:
            weights = 2 * np.random.random((self.input_nodes, nodes)) - 1
        elif len(self.layers) > 0:
            weights = 2 * np.random.random((self.layers[-1].nodes, nodes)) - 1
        else:
            raise AttributeError('There are no layers present.')

        self.layers.append(Layer(nodes, weights, biases))

    def predict(self, X):

        amount_of_layers = len(self.layers)

        for i in range(amount_of_layers):
            layer = self.layers[i]
            if i == 0:
                layer.activations = sigmoid(np.dot(X, layer.weights))
            elif i > 0:
                layer.activations = sigmoid(layer.biases + np.dot(self.layers[i-1].activations, layer.weights))

        return np.round(self.layers[-1].activations)

    def train(self, X, y, alpha, iterations):

        y = np.array([y])

        for iter in range(iterations):

            amount_of_layers = len(self.layers)

            for i in range(amount_of_layers):
                layer = self.layers[i]
                if i == 0:
                    layer.activations = sigmoid(np.dot(X, layer.weights))
                elif i > 0:
                    layer.activations = sigmoid(layer.biases + np.dot(self.layers[i - 1].activations, layer.weights))

            overall_error = np.subtract(y, self.layers[-1].activations)
            print('Overall error is ', overall_error)

            for i in range(amount_of_layers):

                # @TODO This one is fairly tricky
                layer = self.layers[amount_of_layers - 1 - i]
                prev_layer = self.layers[amount_of_layers - 2 - i]

                # @TODO The error is not relative
                deltas = np.multiply(overall_error, derivative_of_sigmoid(layer.activations)) * alpha

                if i == (amount_of_layers - 1):
                    update_weights = X.T.dot(deltas)

                else:
                    update_weights = prev_layer.activations.T.dot(deltas)

                layer.weights += update_weights


class Layer:
    def __init__(self, nodes, weights, biases):
        self.nodes = nodes
        self.weights = weights
        self.biases = biases


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def derivative_of_sigmoid(x):
    return x * (1 - x)

