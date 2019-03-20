import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import os

LEARNING_RATE = 0.001
checkpoint_path = './model/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

train_x = train.iloc[:, 1:].values.astype('float32')
train_y = train.iloc[:, 0].values.astype('int32')

test_x = test.iloc[:, :].values.astype('float32')

# Reshape the features to be 28 x 28
train_x = train_x.reshape(train_x.shape[:1] + (28, 28, 1))
test_x = test_x.reshape(test_x.shape[:1] + (28, 28, 1))

# Transform classes into one hot vectors
train_y = keras.utils.to_categorical(train_y)

# Calculate number of classes
num_classes = len(class_names)

# Normalizing the pixel values to be between 0 and 1
train_x = train_x / 255
test_x = test_x / 255

# Create callbacks for KERAS
callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, min_delta=0, monitor='val_loss')]
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# Set up sneak preview of samples
# plt.figure()
# plt.imshow(train_x[0].reshape(28, 28))
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_x[i].reshape(28, 28), cmap=plt.cm.binary)
#     plt.xlabel(class_names[np.argmax(train_y[i])])
# plt.show()

model = keras.Sequential([

    # Start = 1@28x28
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), strides=(1, 1), activation='relu'),
    # Output = 32@26x26
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    # Output = 32@13x13
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    # Output = 64@11, 11
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    # Output = 64@5x5

    # Flatten into single layer
    keras.layers.Flatten(),
    # Output @1600

    keras.layers.Dense(128, activation=tf.nn.relu),
    # Output @128

    # Dropout 20% of the nodes to help with overfitting
    keras.layers.Dropout(0.2),

    # Connect to output classes
    keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.load_weights(checkpoint_path)
# model.fit(
#     x=train_x,
#     y=train_y,
#     batch_size=32,
#     epochs=1,
#     verbose=1,
#     callbacks=[cp_callback],
#     validation_split=0.05,
#     shuffle=True
# )

predictions = model.predict(test_x)


def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')


def plot_image(i, predictions_array, img):
    img = img.reshape(img.shape[0] ,28, 28)
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{} - prob:{:2.0f}%".format(class_names[predicted_label], 100*np.max(predictions_array)), color='red')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_x)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions)
plt.show()
