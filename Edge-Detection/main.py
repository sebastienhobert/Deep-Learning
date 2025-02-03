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
# filters <=> convolution filter (4D tensor) also called kernel, which is used to detect patterns in the image
# strides <=> stride (integer or list), defines how much the filter moves across the image
# padding <=> padding type ('VALID' or 'SAME') since we use 'VALID' it doesn't add extra borders, it will highlight the edges in the original image
# ==> 'VALID' means no padding (output size will shrink)
# ==> 'SAME' would add zero padding so that the output size remains the same as in the input

# Convert result to numpy for visualization
output_image = conv_layer.numpy().squeeze()
# conv_layer.numpy()
# ==> converts the TensorFlow tensor (result of tf.nn.conv2d()) into a NumPy array
# ==> This allows further processing with NumPy functions or visualization with Matplotlib

# .squeeze()
# ==> removes dimensions with size 1 (like batch and channel dimensions)
# ==> if the output tensor has shape (1, 3, 3, 1) (batch_size=1, height=3, width=3, channels=1),
# ==> .squeeze() will reshape it to (3,3) (removing batch and channels dimensions)
# ==> this makes it easier to visualize the image as a 2D array

# Show the original and filtered images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image.squeeze(), cmap='gray')
# plt.subplot(1, 2, 1) == creates the first subplot in a 1-row, 2-column grid
# plt.imshow(image.squeeze(), cmap='gray')
# ==> .squeeze() removes unnecessary dimensions (from (1, 5, 5, 1) to (5,5))
# ==> cmap='gray' ensures the image is displayed in grayscale

plt.subplot(1, 2, 2)
plt.title("After Convolution")
plt.imshow(output_image, cmap='gray')
# plt.subplot(1, 2, 2) == creates the second subplot
# plt.imshow(output_image, cmap='gray') == displays the filtered image

plt.show()