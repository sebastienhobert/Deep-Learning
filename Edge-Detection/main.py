import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a 5x5 grayscale image
image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2]
], dtype=np.float32)

# Reshape to match TensorFlow input format (batch, height, width, channel)
image = image.reshape(1, 5, 5, 1) # reshaping the above NumPy array called "image" into a 4D tensor
# 1 == batch size (a single image)
# 5 == height
# 5 == width
# 1 == number of channels (grayscale)

# Define a 3x3 filter
filter = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32).reshape(3, 3, 1, 1)
# 3,3 == kernel size (height x width)
# 1 == input channels (for grayscale images, we only have one channel)
# 1 == output channels (defines how many different feature maps this filter generates)

# why reshape? because deep learning libraries expect convolutional filters to be in the format:
# (filter_height, filter_width, input_channels, output_channels)
# this allows for applying multiple filters to multi-channel images (e.g., RGB with 3 input channels)

# Apply convolution
conv_layer = tf.nn.conv2d(image, filters=filter, strides=1, padding='VALID')
# tf.nn.conv2d == performs a 2D convolution operation, typically used in deep learning for feature extraction
# input <=> input image (4D tensor)
# filters <=> convolution filter (4D tensor)
# strides <=> stride (integer or list)
# padding <=> padding type ('VALID' or 'SAME')