#Script to export the model to pngs in the export folder

import sys
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load test data
test_data = np.loadtxt('model/sign_mnist_test.csv', skiprows=1, delimiter=',')

# Split the data, the first column is the label we are training for
# Images are in the format of 28x28 pixels
test_images = test_data[:, 1:]
test_labels = test_data[:, 0]

#Export all images to raw bmp
for i in range(0, test_images.shape[0]):
    #Images are saved as linear arrays, reshape to 28x28
    temp = test_images[i].reshape(28, 28)
    #Export
    plt.imsave('export/' + str(i) + '.png', temp, cmap='gray')
