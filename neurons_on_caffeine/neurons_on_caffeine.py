import numpy as np
import random


class Model:
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.layers = []

    def add_layer(self, nodes):
        biases = np.random.random((nodes, 1))
        if len(self.layers) == 0:
            weights = 2 * np.random.random((nodes, self.input_nodes)) - 1
        elif len(self.layers) > 0:
            weights = 2 * np.random.random((nodes, self.layers[-1].nodes)) - 1
        else:
            raise AttributeError('There are no layers present.')

        self.layers.append(Layer(nodes, weights, biases))

    def train(self, X, y, iterations=50000, alpha=0.1):

        self.learning_rate = alpha

        for i in range(iterations):
            index = random.randint(0, len(X) - 1)
            sample_X = X[index]
            sample_y = y[index]
            self.feedforward(sample_X)
            self.backprop(sample_X, sample_y)

    def predict(self, X):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.activations = sigmoid(np.add(np.dot(layer.weights, X), layer.biases))
            else:
                layer.activations = sigmoid(np.add(np.dot(layer.weights, self.layers[i - 1].activations), layer.biases))

        return self.layers[-1].activations

    def feedforward(self, X):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.activations = sigmoid(np.add(np.dot(layer.weights, X), layer.biases))
            else:
                layer.activations = sigmoid(np.add(np.dot(layer.weights, self.layers[i - 1].activations), layer.biases))

    # TODO Write this one as short as possible
    def backprop(self, X, y):

        for i in range(len(self.layers)):

            # Current layer in scope
            layer = self.layers[len(self.layers) - i - 1]

            # Previous layer activations = input
            input = self.layers[len(self.layers) - i - 2].activations

            # If its the last layer
            if i == 0:
                layer.error = np.subtract(y, layer.activations)

            # If its the first layer
            elif i == len(self.layers) - 1:
                input = X
                next_layer = self.layers[len(self.layers) - i]
                layer.error = np.dot(next_layer.weights.T, next_layer.error)

            # All the other iterations
            else:
                next_layer = self.layers[len(self.layers) - i]
                layer.error = np.dot(next_layer.weights.T, next_layer.error)

            output_derivative = derivative_of_sigmoid(layer.activations)
            delta_weights = self.learning_rate * layer.error * output_derivative * input.T
            delta_biases = self.learning_rate * layer.error * output_derivative
            layer.weights += delta_weights
            layer.biases += delta_biases


class Layer:
    def __init__(self, nodes, weights, biases):
        self.nodes = nodes
        self.weights = weights
        self.biases = biases


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def derivative_of_sigmoid(x):
    output = x * (1 - x)
    return output
