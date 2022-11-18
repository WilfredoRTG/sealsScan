import tensorflow as tf
import numpy as np
import os
from keras.models import load_model

class_names = ['agua','focas','rocas']
model = load_model('model.h5', compile=False)
PATH_TO_DATASET = "testImages/"
PATH_TO_FRAGMENTS = 'cropImages2/'

img_height = 180
img_width = 180
imagePath = "image29/"


model = load_model('model.h5', compile=False)

arrayFocas = []

# for image in os.listdir(PATH_TO_FRAGMENTS):
for imageFragment in os.listdir(PATH_TO_FRAGMENTS + imagePath):
    img = tf.keras.utils.load_img(
        PATH_TO_FRAGMENTS + imagePath + "/" + imageFragment, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    confidence_score = np.max(score)
    if ((class_names[np.argmax(score)]) == "focas") and confidence_score>0.8:
        arrayFocas.append(score)
        print(imageFragment, "es una foca con un", 100 * confidence_score, "% de certeza")

    if ((class_names[np.argmax(score)]) == "focas") and (confidence_score>0.5 and confidence_score<0.8):
        arrayFocas.append(score)
        print(imageFragment, "es una foca con un", 100 * confidence_score, "% de certeza")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

np.save('redAntigua.npy', arrayFocas)
