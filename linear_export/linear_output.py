import tensorflow as tf
import numpy as np

#Load h5 model
model = tf.keras.models.load_model('linear_model.h5')

#Test for value 1 to 10
for i in range(1, 11):
    x_test = np.array([i])
    predicted_y = model.predict(x_test)
    print(predicted_y)
