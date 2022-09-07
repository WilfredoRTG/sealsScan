import cv2
import os

PATH_TO_DATASET = 'dataSet/'
PATH_TO_OUTPUT = 'cropImages/'
SIZE_OF_FRAGMENT = 500

# Read image
def cropImage():
    for index, filename in enumerate(os.listdir(PATH_TO_DATASET)):
        pathOutput = PATH_TO_OUTPUT + 'image' + str(index) + '/'
        os.mkdir(pathOutput)
        fragment(pathOutput, filename)

def fragment(pathOutput, filename):
    img = cv2.imread(PATH_TO_DATASET + filename)
    # Where we want to start and finish in row and column
    width, height, channels = img.shape
    i = 0
    j = 0
    fragment = 0
    while i<width:
        while j<height:
            fragment += 1
            crop_img = img[i:i+SIZE_OF_FRAGMENT, j:j+SIZE_OF_FRAGMENT]
            cv2.imwrite(pathOutput + 'fragment' + str(fragment) + '.jpg', crop_img)
            j += SIZE_OF_FRAGMENT
        i += SIZE_OF_FRAGMENT
        j = 0

cropImage()