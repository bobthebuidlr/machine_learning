import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

with np.load('data/mnist.npz') as d:

    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']

    x_train = tf.keras.utils.normalize(x_train)
    y_train = tf.keras.utils.normalize(y_train)

    plt.imshow(x_train[0], cmap = plt.cm.binary)
    plt.show()


def show_image(img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

