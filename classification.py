import tensorflow as tf
import numpy as np
import os
import keras

class_names = ['agua','foca','rocas']
model = keras.models.load_model('model.h5')
# PATH_TO_DATASET = "testImages/"
PATH_TO_FRAGMENTS = 'cropImages/'

# batch_size = 32
img_height = 180
img_width = 180
image = "image32/"
# for image in os.listdir(PATH_TO_FRAGMENTS):
for imageFragment in os.listdir(PATH_TO_FRAGMENTS + image):
    img = tf.keras.utils.load_img(
        PATH_TO_FRAGMENTS + image + "/" + imageFragment, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    if (class_names[np.argmax(score)]) == "foca":
        print(imageFragment, "es una foca con un", 100 * np.max(score), "% de certeza")
    # break
