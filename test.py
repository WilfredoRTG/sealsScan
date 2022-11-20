import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

class_names = ['Rocas','Focas','Agua']
# model = keras.models.load_model('keras_model.h5', compile=False)
# PATH_TO_FRAGMENTS = 'cropImages2/'
PATH_TO_FRAGMENTS = 'image29Filter/'

img_height = 224
img_width = 224
# imagePath = PATH_TO_FRAGMENTS + "image29/"
imagePath = PATH_TO_FRAGMENTS 
PATH_TO_RESULTS = "ResultsPre-trained/resultsFilter/"
PATH_TO_REVIEW = "ResultsPre-trained/reviewsFilter/"

array = []

model = load_model('keras_model2.h5', compile=False)
data = np.ndarray(shape=(1, img_height, img_width, 3), dtype=np.float32)
count = 1

# for imageFragment in os.listdir(PATH_TO_FRAGMENTS + imagePath):
for imageFragment in os.listdir(imagePath):
    # path = PATH_TO_FRAGMENTS + imagePath + imageFragment
    path = imagePath + imageFragment
    img = cv2.imread(path)

    # image = Image.open(PATH_TO_FRAGMENTS + image  + "fragment" + str(i) + ".jpg").convert('RGB')
    # Replace this with the path to your image
    # print(PATH_TO_FRAGMENTS + image  + "fragment" + str(i) + ".jpg")
    image = Image.open(path).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (img_width, img_width)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # if(class_name=="Focas" and confidence_score > 0.8):
    #     array.append(confidence_score)
    #     pathToResults = PATH_TO_RESULTS_OWN + imageFragment
    #     # cv2.imwrite(pathToResults, img)
    #     print("foca")
    #     print(imageFragment, "es una foca con un",  confidence_score, "% de certeza")

    # if(class_name=="Focas" and (confidence_score > 0.5 and confidence_score < 0.8)):
    #     array.append(confidence_score)
    #     pathToReview = PATH_TO_REVIEW_OWN + imageFragment
    #     # cv2.imwrite(pathToReview, img)
    #     print("foca review")
    #     print(imageFragment, "es una foca con un", confidence_score, "% de certeza")
    # count += 1

    if(class_name=="Focas" and confidence_score > 0.8):
        array.append(confidence_score)
        pathToResults = PATH_TO_RESULTS + imageFragment
        cv2.imwrite(pathToResults, img)
        print(imageFragment, "es una foca con un", 100* confidence_score, "% de certeza")

    if(class_name=="Focas" and (confidence_score > 0.5 and confidence_score < 0.8)):
        array.append(confidence_score)
        pathToReview = PATH_TO_REVIEW + imageFragment
        cv2.imwrite(pathToReview, img)
        print(imageFragment, "es una foca con un", 100* confidence_score, "% de certeza")
    count += 1

np.save('redNueva.npy', array)