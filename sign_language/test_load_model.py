import tensorflow as tf
import numpy as np
# Load the data from csv files
test_data = np.loadtxt('model/sign_mnist_test.csv', skiprows=1, delimiter=',')
print('Train data shape:', test_data.shape)



# Load tf model
model = tf.keras.models.load_model('model/sign_language_model.h5')
# Predict test data and calculate accuracy
test_images = test_data[:, 1:]
test_labels = test_data[:, 0]
test_images = test_images / 255.0
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)