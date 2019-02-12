import matrix as mat
import random


def setup():
    nn = mat.NeuralNetwork(2, 2, 1, 0.1)

    training_data = [[[0, 0], [0]], [[1, 1], [0]], [[1, 0], [1]], [[0, 1], [1]]]

    for i in range(100000):
        item = random.choice(training_data)
        nn.train(item[0], item[1])

    print(
        nn.feedforward([0, 1]),
        nn.feedforward([1, 0]),
        nn.feedforward([1, 1]),
        nn.feedforward([0, 0])
    )


setup()
