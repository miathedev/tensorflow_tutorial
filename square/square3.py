import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([
    keras.layers.Dense(units=16, input_shape=[1], activation='relu'),
    keras.layers.Dense(units=16, activation='relu'), #NOTE: Show add 6 more layers
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1)
])

#Quit if loss is not decreasing
from tensorflow import keras

early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Define the loss and optimizer
model.compile(optimizer='adam', loss='mean_squared_error')


xs = np.arange(-60, 60, 1)
ys = xs * xs

# Train the model
model.fit(xs, ys, epochs=1000, 
    validation_data=(xs, ys),
    callbacks=[early_stopping]
)

import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig('square3_loss_vs_epoch.png')

#Plot training vs prediction
plt.clf()
plt.plot(xs, ys, 'bo', label='Training data')
plt.plot(xs, model.predict(xs), 'r', label='Prediction')
plt.legend()
plt.savefig('square3_training_vs_prediction.png')
# Predict the value
x_test = np.array([4.0])
predicted_y = model.predict(x_test)
print(predicted_y)
