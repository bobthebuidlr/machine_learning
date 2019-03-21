import numpy as np
from termcolor import colored


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_of_sigmoid(x):
    return x * (1 - x)


np.random.seed(0)

# DECIDE ON PARAMETERS
ITERATIONS = 10000
LEARNING_RATE = 0.1
BINARY_DIMENSIONS = 8
INPUT_DIMENSIONS = 2
HIDDEN_DIMENSIONS = 16
OUTPUT_DIMENSIONS = 1

# Preparing labels
LARGEST_NUMBER = np.power(2, BINARY_DIMENSIONS)
binary = np.unpackbits(
    np.array([range(LARGEST_NUMBER)], dtype=np.uint8).T, axis=1
)

int2binary = {}
for i in range(LARGEST_NUMBER):
    int2binary[i] = binary[i]

# Structure of the network
syn_0 = 2 * np.random.random((INPUT_DIMENSIONS, HIDDEN_DIMENSIONS)) - 1
syn_1 = 2 * np.random.random((HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS)) - 1
syn_2 = 2 * np.random.random((HIDDEN_DIMENSIONS, OUTPUT_DIMENSIONS)) - 1

# Update weights
syn_0_update = np.zeros_like(syn_0)
syn_1_update = np.zeros_like(syn_1)
syn_2_update = np.zeros_like(syn_2)

# Training
for i in range(ITERATIONS):

    a_int = np.random.randint(LARGEST_NUMBER / 2)
    a = binary[a_int]

    b_int = np.random.randint(LARGEST_NUMBER / 2)
    b = binary[b_int]

    c_int = a_int + b_int
    c = binary[c_int]

    d = np.zeros_like(c)

    overall_error = 0

    layer_2_deltas = list()
    recurrent_values = list()
    recurrent_values.append(np.zeros(HIDDEN_DIMENSIONS))

    for position in range(BINARY_DIMENSIONS):

        X = np.array([[a[BINARY_DIMENSIONS - position - 1], b[BINARY_DIMENSIONS - position - 1]]])
        y = np.array([[c[BINARY_DIMENSIONS - position - 1]]]).T

        layer_1 = sigmoid(np.dot(X, syn_0) + np.dot(recurrent_values[-1], syn_1))

        layer_2 = sigmoid(np.dot(layer_1, syn_2))

        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * derivative_of_sigmoid(layer_2))
        overall_error += np.abs(layer_2_error[0])

        d[BINARY_DIMENSIONS - position - 1] = np.round(layer_2[0][0])
        recurrent_values.append(layer_1)

    layer_1_deltas_future = np.zeros(HIDDEN_DIMENSIONS)

    # Back propagation
    for position in range(BINARY_DIMENSIONS):

        X = np.array([[a[position], b[position]]])
        layer_1 = recurrent_values[- position - 1]
        layer_1_previous = recurrent_values[- position - 2]

        layer_2_delta = layer_2_deltas[- position - 1]

        layer_1_deltas = (layer_1_deltas_future.dot(syn_1.T) + layer_2_delta.dot(syn_2.T)) * derivative_of_sigmoid(layer_1)

        syn_2_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        syn_1_update += np.atleast_2d(layer_1_previous).T.dot(layer_1_deltas)
        syn_0_update += X.T.dot(layer_1_deltas)

        layer_1_deltas_future = layer_1_deltas

    syn_0 += syn_0_update * LEARNING_RATE
    syn_1 += syn_1_update * LEARNING_RATE
    syn_2 += syn_2_update * LEARNING_RATE

    syn_0_update *= 0
    syn_1_update *= 0
    syn_2_update *= 0

    if i % 1000 == 0:

        a = np.packbits(a)
        b = np.packbits(b)
        c = np.packbits(c)
        d = np.packbits(d)

        print("Input was: %s + %s" % (a, b))
        print("Actual sum is %s" % c)
        if c == d:
            print(colored(("Guess was: %s" % d), 'green'))

        else:
            print(colored(("Guess was: %s" % d), 'red'))

        print('\n')





