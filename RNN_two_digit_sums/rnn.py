import copy, numpy as np

np.random.seed(3)


# Define sigmoid function
def sigmoid(x):
    output = 1/(1 + np.exp(-x))
    return output


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
iterations = 1

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

    # Moving along the 8 positions of the binary numbers
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
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_deltas = np.zeros(hidden_dim)

