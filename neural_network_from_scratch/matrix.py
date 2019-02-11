import random
import math


# Standard sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Derivative of the sigmoid (y is expected to already be activated)
def dsigmoid(y):
    return y * (1 - y)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Construct the layers and learning rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize the weight matrices
        self.l1_weights = Matrix(self.hidden_nodes, self.input_nodes)
        self.l2_weights = Matrix(self.output_nodes, self.hidden_nodes)
        self.l1_weights.randomize()
        self.l2_weights.randomize()

        # Initialize the bias matrices
        self.l1_bias = Matrix(self.hidden_nodes, 1)
        self.l2_bias = Matrix(self.output_nodes, 1)
        self.l1_bias.randomize()
        self.l2_bias.randomize()

    def feedforward(self, input_array):

        # Generate hidden activations
        inputs = Matrix.from_array(input_array)
        l1_activations = Matrix.dot_product(self.l1_weights, inputs)
        l1_activations.add(self.l1_bias)
        l1_activations.map(sigmoid)

        # Generate output activations
        l2_activations = Matrix.dot_product(self.l2_weights, l1_activations)
        l2_activations.add(self.l2_bias)
        l2_activations.map(sigmoid)

        # Sending back to the caller
        return l2_activations.to_array()

    def train(self, input_array, target_array):

        # Generate hidden activations
        inputs = Matrix.from_array(input_array)
        l1_activations = Matrix.dot_product(self.l1_weights, inputs)
        l1_activations.add(self.l1_bias)
        l1_activations.map(sigmoid)

        # Generate output activations
        l2_activations = Matrix.dot_product(self.l2_weights, l1_activations)
        l2_activations.add(self.l2_bias)
        l2_activations.map(sigmoid)

        # Convert the targets to a matrix
        targets = Matrix.from_array(target_array)

        # Calculate layer 2 errors
        # ERROR = TARGETS - L2_ACTIVATIONS
        l2_errors = Matrix.subtract(targets, l2_activations)

        # Calculate the gradients
        l2_gradients = l2_activations
        l2_gradients.map(dsigmoid)
        l2_gradients.scale(l2_errors)
        l2_gradients.scale(self.learning_rate)

        # Transpose activations of L1
        l1_activations_t = Matrix.transpose(l1_activations)
        l2_deltas = Matrix.dot_product(l2_gradients, l1_activations_t)

        # Adjust the weights of Layer 2
        self.l2_weights.add(l2_deltas)
        self.l2_bias.add(l2_gradients)

        # Calculate layer 1 errors
        l2_weights_t = Matrix.transpose(self.l2_weights)
        l1_errors = Matrix.dot_product(l2_weights_t, l2_errors)

        # Calculate the gradients
        l1_gradients = l1_activations
        l1_gradients.map(dsigmoid)
        l1_gradients.scale(l1_errors)
        l1_gradients.scale(self.learning_rate)

        # Transpose activations of L0
        l0_activations_t = Matrix.transpose(inputs)
        l1_deltas = Matrix.dot_product(l1_gradients, l0_activations_t)

        # Adjust the weights of Layer 2
        self.l1_weights.add(l1_deltas)
        self.l1_bias.add(l1_gradients)


class Matrix:
    # Initialize the matrix with input columns and rows
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []

        for row in range(rows):
            self.data.append([])
            for _ in range(cols):
                self.data[row].append(0)

    # Print out matrix row by row
    def print(self):
        for row in range(self.rows):
            print(self.data[row])
        print('\n')

    # Create random numbers between given range
    def randomize(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = random.uniform(-1, 1)

    # Add number to every element
    def add(self, n):
        if isinstance(n, Matrix):
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] += n.data[row][col]
        else:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] += n

    # Multiply data by scalar
    def scale(self, n):
        if isinstance(n, Matrix):
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] *= n.data[row][col]
        else:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] *= n

    # Apply function to every element of the matrix
    def map(self, fn):
        for row in range(self.rows):
            for col in range(self.cols):
                val = self.data[row][col]
                self.data[row][col] = fn(val)

    def to_array(self):
        arr = []
        for row in range(self.rows):
            for col in range(self.cols):
                arr.append(self.data[row][col])
        return arr

    # Transpose the matrix
    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.cols, matrix.rows)
        for col in range(matrix.cols):
            for row in range(matrix.rows):
                result.data[col][row] = matrix.data[row][col]

        return result

    # Matrix dot product of two different input matrices
    @staticmethod
    def dot_product(m1, m2):
        if isinstance(m2, Matrix) and m1.cols == m2.rows:
            m3 = Matrix(m1.rows, m2.cols)
            product = 0
            for row in range(m1.rows): # 0 and 1
                for col in range(m2.cols): # 0 and 1
                    for row_n in range(m1.cols): # 0, 1, 2
                        product += m1.data[row][row_n] * m2.data[row_n][col]
                    m3.data[row][col] = product
                    product = 0
            return m3
        elif isinstance(m2, Matrix) and m2.cols == m1.rows:
            m3 = Matrix(m2.rows, m1.cols)
            product = 0
            for row in range(m2.rows): # 0 and 1
                for col in range(m1.cols): # 0 and 1
                    for row_n in range(m2.cols): # 0, 1, 2
                        product += m2.data[row][row_n] * m1.data[row_n][col]
                    m3.data[row][col] = product
                    product = 0
            return m3
        else:
            raise ValueError('Dot product can not be performed. Number of columns and rows are not reversed.')

    @staticmethod
    def from_array(arr):
        m = Matrix(len(arr), 1)
        for i in range(m.rows):
            m.data[i][0] = arr[i]
        return m

    @staticmethod
    def subtract(inputs, targets):
        result = Matrix(inputs.rows, inputs.cols)
        for row in range(result.rows):
            for col in range(result.cols):
                result.data[row][col] = inputs.data[row][col] - targets.data[row][col]
        return result
