'''
Step 1: Import Libraries
'''
import tensorflow as tf
# tensorflow == open-source machine learning framework

from tensorflow.keras import layers, models
# tensorflow.keras == API for building neural networks
# layers == this module provides building blocks for deep learning models such as convolutional layers (Conv2D) or fully connected layers (Dense)
# models == this module is used for defining and managing neural network architectures such as sequential models (Sequential())

import matplotlib.pyplot as plt
# matplotlib.pyplot == Python library for data visualization
# plt == to display images (visualizing samples from the MNIST dataset)
# plt == to track model training performance using loss/accuracy curves

'''
Step 2: Load the MNIST Dataset
'''
mnist = tf.keras.datasets.mnist
# tf.keras.datasets.mnist == module in TensorFlow that provides access to the MNIST dataset consisting of:
# 70000 grayscale images of handwritten digits (0-9)
# ==> 60000 images for training
# ==> 10000 images for testing

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train == training images (60000 samples, each 28x28 pixels)
# y_train == corresponding labels (digits 0-9) for the training images
# x_test == testing images (10000 samples, each 28x28 pixels)
# y_test == corresponding labels for the test images

x_train, x_test = x_train / 255.0, x_test / 255.0
# dividing by 255.0 normalizes all pixel values to be between 0 and 1
# x_train and x_test == NumPy arrays representing grayscale or RGB image datasets

'''
Step 3: Build the Neural Network Model
'''
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # flatten the 28x28 image into a 784-dimensional vector
    layers.Dense(128, activation='relu'), # fully connected layer with 128 units and ReLU activation
    layers.Dropout(0.2), # drops 20% of the neurons to prevent overfitting
    layers.Dense(10, activation='softmax') # output layer with 10 units (for 10 digits), softmax function ensures that the sum of all inputs is 1
])

'''
Step 4: Compile the Model
'''
model.compile(optimizer='adam', # adaptive moment estimation, it adjusts weights and learning rates, so it reduces loss and improve accuracy
              loss='sparse_categorical_crossentropy', # used because the target labels are provided as integers instead of one-hot encoded vectors
              metrics=['accuracy']) # tracks accuracy during training and testing

'''
Step 5: Train the Model
'''
print("Training the model..")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# history == it stores training metrics (loss, accuracy) which can be visualized
# x_train, y_train == training dataset (features and labels)
# epochs=5 == the model trains for 5 complete passes over the dataset
# validation_data=(x_test, y_test) == evaluates the model using test data after each epoch

'''
Step 6: Evaluate the Model
'''
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# x_test, y_test == the test dataset (features & labels)
# verbose=2 == controls the output format
# ==> 0: silent mode (no output format)
# ==> 1: progress bar (default)
# ==> 2: one line per epoch

print(f"\nTest accuracy: {test_acc:4f}")

'''
Step 7: Visualize Training Results
'''
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


