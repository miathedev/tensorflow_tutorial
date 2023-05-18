# Sign language recognition
# Model url: https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset

import sys
import matplotlib.pyplot as plt
import string
import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2  # For future use

# Load the data from csv files
train_data = np.loadtxt('model/sign_mnist_train.csv',
                        skiprows=1, delimiter=',')
test_data = np.loadtxt('model/sign_mnist_test.csv', skiprows=1, delimiter=',')
print('Train data shape:', train_data.shape)
# Split the data, the first column is the label we are training for
# Images are in the format of 28x28 pixels
train_images = train_data[:, 1:]
train_labels = train_data[:, 0]

#Reshape train_images to 28x28
train_images = train_images.reshape(train_images.shape[0], 28, 28)

test_images = test_data[:, 1:]
test_labels = test_data[:, 0]

#Reshape test_images to 28x28
test_images = test_images.reshape(test_images.shape[0], 28, 28)

# Normalize the data, dif. term in pipeline, (backprop algo.) -> weight multip, with x>1 way to over sizeing in opt. proc.
train_images = train_images / 255.0
test_images = test_images / 255.0

test_images = test_images.reshape((-1, 28, 28, 1))
train_images = train_images.reshape((-1, 28, 28, 1))


# Build the model, feed the images into a 64x64x1 input layer
# NOTE: Adding more layers actually makes it worse...
# https://keras.io/api/layers/activations/
# model = keras.Sequential([
#     # selu: scaled exponential linear unit
#     keras.layers.Dense(64, activation='selu', input_shape=(784,)),
#     # sigmoid: Sigmoid activation function
#     keras.layers.Dense(64, activation='sigmoid'),
#     # softmax for probability distribution
#     keras.layers.Dense(25, activation='softmax')
# ])

#Print image shape
print('Image shape:', train_images.shape);
#exit
#sys.exit()
model = keras.Sequential([
    #Use Conv2D to extract features from the input image
    #https://keras.io/api/layers/convolution_layers/convolution2d/
    #Images are saved in 27455, 28, 28 format
    #Conv2D: 2D convolution layer (e.g. spatial convolution over images).

    #Pooling, wie groß kann die Bildgröße auf evol. Handy sein
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2),strides=2),
    
    keras.layers.Dropout(0.2), # Dropping random 2% data out for learning variety.
    
    keras.layers.Conv2D(32,(3,3),padding="same",activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2),strides=2),
    
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(16,(3,3),padding="same",activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(25,activation="softmax")

])

# Compile the model
#https://keras.io/api/optimizers/
#https://keras.io/api/losses/
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model until it doesnt improve anymore
# NOTE: When overfitting, generalisation: from unseen data is getting worse
model.fit(train_images, train_labels, epochs=1000000000000, 
          validation_data=(test_images, test_labels), 
          callbacks=[keras.callbacks.EarlyStopping(patience=10, min_delta=0.000001, restore_best_weights=True)])


# Save the model
model.save('model/sign_language_model.h5')

plt.plot(model.history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig('loss_vs_epoch.png')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# Group all images by label
images_by_label = {}
for i in range(len(test_labels)):
    label = test_labels[i]
    if label not in images_by_label:
        images_by_label[label] = []
    images_by_label[label].append(test_images[i])

# sort labels by label value
images_by_label = dict(sorted(images_by_label.items()))

# print labels count
# for label in images_by_label:
#    #print(label, ':', len(images_by_label[label]))
#    #calculate accuracy for each label
#    label_test_images = np.array(images_by_label[label])
#    label_test_labels = np.full(len(label_test_images), label)
#    test_loss, test_acc = model.evaluate(label_test_images, label_test_labels)
#    print('Test accuracy for label', label, ':', test_acc)

# Create graph, showing accuracy for each label
x = images_by_label.keys()

y = []
for label in images_by_label:
    label_test_images = np.array(images_by_label[label])
    label_test_labels = np.full(len(label_test_images), label)
    test_loss, test_acc = model.evaluate(label_test_images, label_test_labels)
    y.append(test_acc)

#model_string = ''
#for layer in model.layers:
#    config = layer.get_config()
#    model_string += str(config['units']) + ', act: ' + \
#        str(config['activation']) + '\n'
#print(model_string)

#Log to result.txt
with open('result.txt', 'a') as f:
    f.write('=============================================================\n')
    f.write('Test accuracy: ' + str(test_acc) + '\n')
    f.write('Test loss: ' + str(test_loss) + '\n')
    f.write('Accuracy for each label: ' + str(y) + '\n')
 #   f.write('Model: ' + model_string + '\n')
# ==============================================================================
plt.close('all')
plt.figure(figsize=(10, 5))
# Generate A to Z labels
x_ticks = [f"{string.ascii_uppercase[i]}" for i in range(0, 26)]

print(x_ticks)
plt.bar(x, y)
plt.xticks(np.arange(len(x_ticks)), x_ticks)
plt.title('Accuracy for each label')
plt.ylabel('Accuracy')

# Add model string to graph, add extra whitespace to make it fit under x-axis
#plt.text(0, 0.1, model_string, bbox=dict(facecolor='red', alpha=0.8))

# Add median line
median = np.median(y)
plt.axhline(y=median, color='r', linestyle='-')
plt.text(0, median, 'Median: {:.2f}'.format(
    median), bbox=dict(facecolor='red', alpha=0.5))


plt.xlabel('J=9, Z=25 is missing in the dataset cause of gesture motion\n')
plt.savefig('accuracy_for_each_label.png')


# ==============================================================================
# Save the model as a tflite file
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#open("model/sign_language_model.tflite", "wb").write(tflite_model)

# Save example image from dataset
cv2.imwrite('example_image.png', train_images[0].reshape(28, 28) * 255.0)
