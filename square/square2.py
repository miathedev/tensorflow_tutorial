import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
# NOTE: This shit needs a lot of time. What if we could stop earlier when we are happy with the result?
model = tf.keras.Sequential([
    keras.layers.Dense(units=16, input_shape=[1], activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1)
])

# Define the loss and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')


xs = np.arange(-60, 60, 1)
ys = xs * xs

# Train the model
model.fit(xs, ys, epochs=1000)

import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig('square2_loss_vs_epoch.png')

#Plot training vs prediction
plt.clf()
plt.plot(xs, ys, 'bo', label='Training data')
plt.plot(xs, model.predict(xs), 'r', label='Prediction')
plt.legend()
plt.savefig('square2_training_vs_prediction.png')
# Predict the value
x_test = np.array([4.0])
predicted_y = model.predict(x_test)
print(predicted_y)
