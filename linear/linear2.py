import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([
    keras.layers.Dense(units=16, input_shape=[1]), 
    keras.layers.Dense(units=1)
])

# Define the loss and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate xs for a range from -1 to 100
xs = np.arange(-1, 60, 1) #default: 10, on increasing the range, the loss decreases, but not as much
ys = xs * 2

# Train the model
model.fit(xs, ys, epochs=1000)

import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
#max = 5000
plt.ylim(0, 5000)
plt.savefig('linear2_loss_vs_epoch.png')

#Plot training vs prediction
plt.clf()
plt.plot(xs, ys, 'bo', label='Training data')
plt.plot(xs, model.predict(xs), 'r', label='Prediction')
plt.legend()
plt.savefig('linear2_training_vs_prediction.png')

# Predict the value
x_test = np.array([4.0])
predicted_y = model.predict(x_test)
print(predicted_y)
