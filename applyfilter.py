from utils import applyFilters
import os 
import cv2

PATH_TO_IMAGES = "cropImages2/image29/"
PATH_TO_OUTPUT = "image29Filter/"

count = 0

for image in os.listdir(PATH_TO_IMAGES):
    path = PATH_TO_IMAGES + image
    img = cv2.imread(path)
    imgFilter = applyFilters(img)
    pathOutput = PATH_TO_OUTPUT + str(count) + ".jpg"
    cv2.imwrite(pathOutput, imgFilter)
    count += 1