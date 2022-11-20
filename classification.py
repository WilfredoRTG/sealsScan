import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import cv2
from utils import applyFilters
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

class_names = ['agua', 'focas', 'rocas']
# model = load_model('model.h5', compile=False)

PATH_TO_DATASET = "testImages/"
PATH_TO_FRAGMENTS = 'image29Filter/'
# PATH_TO_FRAGMENTS = 'cropImages2/'
PATH_TO_RESULTS_OWN = "ResultsOwnCNN/resultsFilter/"
PATH_TO_REVIEW_OWN = "ResultsOwnCNN/reviewsFilter/"

img_height = 180
img_width = 180
imagePath = PATH_TO_FRAGMENTS
# imagePath = PATH_TO_DATASET + "foca/"


def classifier(modelToLoad):
    model = load_model(modelToLoad, compile=False)
    y_pred = []
    y_true = []
    arrayFocas = []
    # for image in os.listdir(PATH_TO_FRAGMENTS):
    for imageFragment in os.listdir(imagePath):
        # img = tf.keras.utils.load_img(
        #     PATH_TO_FRAGMENTS + imagePath + "/" + imageFragment, target_size=(img_height, img_width)
        # )
        path = imagePath + "/" + imageFragment
        imgToWrite = cv2.imread(path)
        img = tf.keras.utils.load_img(
            path, target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        confidence_score = np.max(score)

        if ((class_names[np.argmax(score)]) == "focas") and confidence_score > 0.8:
            pathToResults = PATH_TO_RESULTS_OWN + imageFragment
            arrayFocas.append(score)
            cv2.imwrite(pathToResults, imgToWrite)
            print(imageFragment, "es una foca con un",
                  100 * confidence_score, "% de certeza")

        if ((class_names[np.argmax(score)]) == "focas") and (confidence_score > 0.5 and confidence_score < 0.8):
            pathToReview = PATH_TO_REVIEW_OWN + imageFragment
            arrayFocas.append(score)
            cv2.imwrite(pathToReview, imgToWrite)
            print(imageFragment, "es una foca con un",
                  100 * confidence_score, "% de certeza")

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    return y_pred, y_true, class_names
    # np.save('redAntigua.npy', arrayFocas)
    # np.save('redAntiguaRocas.npy', arrayRocas)


'''
Que tengo?
- 3 carpetas con imagenes de focas, rocas y agua con filtros
- Modelo por por probar
'''


def contructorOfCM(modelToLoad, folderOfImages, classNames, height, width, NUMBER_AGUA=10, NUMBER_FOCAS=15, NUMBER_ROCAS=20):

    model = load_model(modelToLoad, compile=False)
    countFocas = 0
    countRocas = 0
    countAgua = 0
    y_true = []
    y_pred = []

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    for folder in os.listdir(folderOfImages):
        for image in os.listdir(folderOfImages + folder):

            path = folderOfImages + folder + "/" + image
            # img = tf.keras.utils.load_img(
            # path, target_size=(height, width)
            # )
            # img_array = tf.keras.utils.img_to_array(img)
            # img_array = tf.expand_dims(img_array, 0) # Create a batch

            # predictions = model.predict(img_array)
            # score = tf.nn.softmax(predictions[0])

            image = Image.open(path).convert('RGB')
            size = (height, width)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (
                image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = classNames[index]

            if folder == classNames[0]:
                y_true.append(classNames[0])
                y_pred.append(class_name)
                countAgua += 1
                if countAgua == NUMBER_ROCAS:
                    break
            elif folder == classNames[1]:
                y_true.append(classNames[1])
                y_pred.append(class_name)
                countFocas += 1
                if countFocas == NUMBER_FOCAS:
                    break
            elif folder == classNames[2]:
                y_true.append(classNames[2])
                y_pred.append(class_name)
                countRocas += 1
                if countRocas == NUMBER_AGUA:
                    break

    cm = confusion_matrix(y_true, y_pred, labels=classNames)
    accuracy_rocas = cm[0][0]/(NUMBER_ROCAS)
    accuracy_focas = cm[1][1]/(NUMBER_FOCAS)
    accuracy_agua = cm[2][2]/(NUMBER_AGUA)
    score = accuracy_score(y_true, y_pred)

    print("Accuracy focas: ", accuracy_focas)
    print("Accuracy rocas: ", accuracy_rocas)
    print("Accuracy agua: ", accuracy_agua)
    print("Accuracy: ", score)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classNames)
    disp.plot()
    plt.show()


contructorOfCM("keras_model2.h5", "testImages/",
               ["rocas", "focas", "agua"], 224, 224)
