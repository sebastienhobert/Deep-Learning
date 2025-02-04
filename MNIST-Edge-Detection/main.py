import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select a sample image (let's take the first one)
image = x_train[0] # first digit image
image = image.astype(np.float32) / 255.0 # normalize pixel values (0 to 1)

# Reshape for TensorFlow (batch, height, width, channels)
image = image.reshape(1, 28, 28, 1)

# Define a simple 3x3 edge detection filter
filter = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32).reshape(3, 3, 1, 1) # height, width, input_channels, output_channels

# Apply convolution
conv_layer = tf.nn.conv2d(image, filters=filter, strides=1, padding='SAME')

# Convert result to numpy for visualization
output_image = conv_layer.numpy().squeeze()

# Show the original and filtered images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original MNIST Image")
plt.imshow(image.squeeze(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("After Convolution (Edges)")
plt.imshow(output_image, cmap='gray')

plt.show()