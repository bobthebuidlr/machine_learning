import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

EPOCHS = 5


def show_image(img):
    plt.imshow(img, cmap="Dark2")
    plt.show()


def define_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_test[0].shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


def train_model(x_train, y_train, x_test, y_test):
    model = define_model()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x=x_train, y=y_train, epochs=EPOCHS)

    model.save('models/3_layer.h5')

    (loss, acc) = model.evaluate(x_test, y_test)

    print('loss: ', loss, '\n', 'accuracy: ', acc)


def load_model():
    model.summary()
    # return model.load_weights('models/3_layer.h5')


with np.load('data/mnist.npz') as d:
    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # train_model(x_train, y_train, x_test, y_test)
    model = tf.keras.models.load_model('models/3_layer.h5')
    model.summary()
    example = x_test[0].reshape(1, 28, 28)
    model.predict(example)


