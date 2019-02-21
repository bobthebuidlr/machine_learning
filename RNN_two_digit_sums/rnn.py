# TODO - FIND THE BUG THAT IS HAMPERING PROPER EXECUTION
# TODO - UNDERSTAND THE ERROR CALCULATIONS PROPERLY

import copy, numpy as np

np.random.seed(0)


# Define sigmoid function
def sigmoid(x):
    output = 1/(1 + np.exp(-x))
    output


# Calculate the derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Training data for binary numbers
int2binary = {}
binary_dim = 8  # Equals 1 byte

largest_number = np.power(2, binary_dim)  # Equals 256 for 1 byte
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1
)
for i in range(largest_number):
    int2binary[i] = binary[i]

# Static environment variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
iterations = 10000

# Initialize the RNN weights for each layer
syn_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
syn_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
syn_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

# Initialize update arrays for the weights
syn_0_update = np.zeros_like(syn_0)
syn_1_update = np.zeros_like(syn_1)
syn_h_update = np.zeros_like(syn_h)


# Training function
for i in range(iterations):
    # Create training examples. A, B for input and output C
    a_int = np.random.randint(largest_number/2)
    a = binary[a_int]

    b_int = np.random.randint(largest_number/2)
    b = binary[b_int]

    c_int = a_int + b_int
    c = binary[c_int]

    # Storage for best guess of output
    d = np.zeros_like(c)

    overall_error = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # Moving along the 8 positions of the binary numbers - FOrward propagation
    for position in range(binary_dim):

        # Generating the input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # Hidden layer calcs
        layer_1 = sigmoid(np.dot(X, syn_0) + np.dot(layer_1_values[-1], syn_h))

        # Output layer calcs
        layer_2 = sigmoid(np.dot(layer_1, syn_1))

        # Error calcs
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_derivative(layer_2))
        overall_error += np.abs(layer_2_error)

        # Construct the binary output per position
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # Copy the layer_1 for use in next iteration
        # TODO deepcopy might not be needed!
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # Backpropagation (starting at the last binary position
    for position in range(binary_dim):

        # Input is the last binary position for both numbers
        X = np.array([[a[position], b[position]]])

        # Selecting the current hidden layer activations
        layer_1 = layer_1_values[-position - 1]

        # Selecting the previous hidden layer activations
        prev_layer_1 = layer_1_values[-position - 2]

        # Get the errors from the y and output
        layer_2_delta = layer_2_deltas[-position - 1]

        # Get the error at the hidden layer by summing the dot product of
        # future layer 1 with hidden weights and layer 2 errors by layer 1 weights
        # and multiplying that with the derivative of the layer 1 activations
        layer_1_deltas = (future_layer_1_delta.dot(syn_h.T) + layer_2_delta.dot(syn_1.T)) * sigmoid_derivative(layer_1)

        # Updating all the weights
        syn_1_update = np.atleast_2d(layer_1).T.dot(layer_2_delta)
        syn_0_update = X.T.dot(layer_1_deltas)
        syn_h_update = np.atleast_2d(prev_layer_1).T.dot(layer_1_deltas)

    # Update the weights
    syn_1 += syn_1_update * alpha
    syn_0 += syn_0_update * alpha
    syn_h += syn_h_update * alpha

    # Empty all the update synapses
    syn_1_update *= 0
    syn_0_update *= 0
    syn_h_update *= 0

    if i % 1000 == 0:
        a = np.packbits(a)
        b = np.packbits(b)
        c = np.packbits(c)
        d = np.packbits(d)

        print("Input was: %s + %s" % (a, b))
        print("Actual sum is %s" % c)
        print("Guess was: %s" % d)
