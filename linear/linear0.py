import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# NOTE: Steps to train and use a model
# 1. Define the model
# 2. Compile the model
# ?. Load the data, can be on step 0 as well
# 3. Train the model
# 4. Predict the value
# 5. Analyze the results to optimize the model

# NOTE: 1. Define the model ====================
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1]),
    keras.layers.Dense(units=1)
])

# NOTE: 2. Compile the model ====================
# Define the loss and optimizer
# The loss is used to calculate the error to minimize
# The optimizer is used to minimize the loss, 
#   it uses the error to calculate the gradient, which is used to update the weights
model.compile(optimizer='adam', loss='mean_squared_error')

# NOTE: ?. Load the data ====================
# Generate xs for a range from -1 to 60
xs = np.arange(-1, 60, 1)
ys = xs * 2

# NOTE: 3. Train the model ====================
# Train the model
model.fit(xs, ys, epochs=1000)

# NOTE: 4. Predict the value ====================
# Predict the value
#x_test = np.array([4.0])
#predicted_y = model.predict(x_test)
#print(predicted_y)

# NOTE: 5. Analyze the results to optimize the model ====================
#Plot training vs prediction

plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
#max = 5000
plt.ylim(0, 5000)
plt.savefig('linear0_loss_vs_epoch.png')

plt.clf()
plt.plot(xs, ys, 'bo', label='Training data')
plt.plot(xs, model.predict(xs), 'r', label='Prediction')
plt.legend()
plt.savefig('linear0_training_vs_prediction.png')


