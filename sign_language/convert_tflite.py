import tensorflow as tf
import numpy as np

#Convert ht5 model to tflite model
model = tf.keras.models.load_model('model/sign_language_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model/sign_language_model.tflite", "wb").write(tflite_model)
