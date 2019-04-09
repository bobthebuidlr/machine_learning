import numpy as np

# Define random seed for reproducability
np.random.seed(3)

# Define input and targets
X = np.array([[[0, 0]], [[1, 0]], [[0, 1]], [[1, 1]]])
y = np.array([[0], [1], [1], [0]])

# Structure of the network
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

# Define the learning rate
learning_rate = 0.15

# Number of iterations
iterations = 50000

# Initialize weights and biases
W1 = np.random.randn(input_nodes, hidden_nodes)
b1 = np.random.randn(1, hidden_nodes)

W2 = np.random.randn(hidden_nodes, output_nodes)
b2 = np.random.randn(1, output_nodes)


# Define the actiation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of activations function
def deriv_sigmoid(x):
    return x * (1 - x)


# Define the feedforward function
def feedforward(X, predict=False):

    hidden_activations = sigmoid(X.dot(W1) + b1)
    output_activations = sigmoid(hidden_activations.dot(W2) + b2)

    if predict:
        print(output_activations)
        return int(np.round(output_activations))

    return hidden_activations, output_activations


# Define backprop function for updating weights
def backpropagation(X, y):

    global W1
    global b1
    global W2
    global b2

    hidden_activations, output_activations = feedforward(X)

    # Calculate the errors at every layer
    output_error = 2 * (y - output_activations)
    hidden_error = output_error * W2.T

    # Calculate the updates for output weights / biases
    b2_update = (output_error * deriv_sigmoid(output_activations)) * learning_rate
    W2_update = hidden_activations.T.dot(b2_update)

    # Calculate the updates for hidden weights / biases
    b1_update = (hidden_error * deriv_sigmoid(hidden_activations)) * learning_rate
    W1_update = X.T.dot(b1_update)

    # Update all respective weights and biases
    W1 += W1_update
    b1 += b1_update
    W2 += W2_update
    b2 += b2_update


def train(iterations):

    for i in range(iterations):
        index = np.random.randint(0, 4)
        backpropagation(X[index], y[index])

    print('Prediction for ', X[0], 'is: ', feedforward(X[0], predict=True))
    print('Prediction for ', X[1], 'is: ', feedforward(X[1], predict=True))
    print('Prediction for ', X[2], 'is: ', feedforward(X[2], predict=True))
    print('Prediction for ', X[3], 'is: ', feedforward(X[3], predict=True))


train(iterations)
