import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL

class_names = ['agua','focas','rocas']
model = load_model('model.h5', compile=False)
PATH_TO_DATASET = "testImages/"
PATH_TO_FRAGMENTS = 'cropImages2/'

img_height = 180
img_width = 180
imagePath = "image29/"


model = load_model('model.h5', compile=False)
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# for imageFragment in os.listdir(PATH_TO_FRAGMENTS + imagePath):
#     path = PATH_TO_FRAGMENTS + imagePath + imageFragment
#     # image = Image.open(PATH_TO_FRAGMENTS + image  + "fragment" + str(i) + ".jpg").convert('RGB')
#     # Replace this with the path to your image
#     # print(PATH_TO_FRAGMENTS + image  + "fragment" + str(i) + ".jpg")
#     image = Image.open(path).convert('RGB')

#     #resize the image to a 224x224 with the same strategy as in TM2:
#     #resizing the image to be at least 224x224 and then cropping from the center
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

#     #turn the image into a numpy array
#     image_array = np.asarray(image)

#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     if(class_name=="Focas" ):
#         print('Class: ', class_name, "Confidence score: ", confidence_score, "Image: ", imageFragment)



arrayFocas = []
# arrayRocas = []

# for image in os.listdir(PATH_TO_FRAGMENTS):
for imageFragment in os.listdir(PATH_TO_FRAGMENTS + imagePath):
    img = tf.keras.utils.load_img(
        PATH_TO_FRAGMENTS + imagePath + "/" + imageFragment, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # index = np.argmax(predictions)
    # confidence_score = predictions[0][index]
    confidence_score = np.max(score)
    if ((class_names[np.argmax(score)]) == "focas") and confidence_score>0.8:
        arrayFocas.append(score)
        print(imageFragment, "es una foca con un", 100 * confidence_score, "% de certeza")

    if ((class_names[np.argmax(score)]) == "focas") and (confidence_score>0.5 and confidence_score<0.8):
        arrayFocas.append(score)
        print(imageFragment, "es una foca con un", 100 * confidence_score, "% de certeza")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

np.save('redAntigua.npy', arrayFocas)
# np.save('redAntiguaRocas.npy', arrayRocas)