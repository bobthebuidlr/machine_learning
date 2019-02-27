import tensorflow as tf
import numpy as np

# Structure of the network:
# 2 input nodes, then 3 hidden nodes, then 1 output node

y = 1.0
X = np.ndarray([[1, 0]])
# X = tf.placeholder(dtype=tf.float64, name="X")

input_nodes = 2
hidden_nodes = 3
output_nodes = 1

hidden_weights = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes], dtype=tf.float64, stddev=0.5), trainable=True)
hidden_biases = tf.Variable(tf.truncated_normal([hidden_nodes, 1], stddev=0.5, dtype=tf.float64), trainable=True)

output_weights = tf.Variable(tf.truncated_normal([hidden_nodes, output_nodes], stddev=0.5, dtype=tf.float64), trainable=True)
output_biases = tf.Variable(tf.truncated_normal([output_nodes, 1], stddev=0.5, dtype=tf.float64), trainable=True)

hidden_activations = tf.sigmoid(tf.matmul(X, hidden_weights) + tf.transpose(hidden_biases))
output_activations = tf.sigmoid(tf.matmul(hidden_activations, output_weights) + tf.transpose(output_biases))

loss = tf.square(y - output_activations, name="loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(1):
        total_loss = 0

        _, l = sess.run([optimizer, loss])


    print(loss.eval())
    writer.close()
