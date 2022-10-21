from scipy import ndimage
import cv2
import os
import numpy as np

PATH_TO_DATASET = 'dataSet/'
PATH_TO_OUTPUT = 'cropImages/'
SIZE_OF_FRAGMENT = 500

# Read images from dataSet folder and crop them by fragments of 500x500px
def cropImage():
    for index, filename in enumerate(os.listdir(PATH_TO_DATASET)):
        pathOutput = PATH_TO_OUTPUT + 'image' + str(index+1) + '/'
        os.mkdir(pathOutput)
        fragment(pathOutput, filename)


def fragment(pathOutput, filename):
    # img = cv2.imread(PATH_TO_DATASET + filename)
    img = cv2.imread(filename)
    # Where we want to start and finish in row and column
    width, height, channels = img.shape
    i = 0
    j = 0
    fragment = 0
    while i < width:
        while j < height:
            fragment += 1
            crop_img = img[i:i+SIZE_OF_FRAGMENT, j:j+SIZE_OF_FRAGMENT]
            cv2.imwrite(pathOutput + 'fragment' +
                        str(fragment) + '.jpg', crop_img)
            j += SIZE_OF_FRAGMENT
        i += SIZE_OF_FRAGMENT
        j = 0


# cropImage()
fragment("./test/image2/", "masked_image2.jpg")
# Image reading
# img = cv2.imread('big_seal.png')


# # ----------------------------------------------- mau ----------------------------------------------- #
# '''
#     1. Invert colors
#     2. Gamma correction with 2 value
#     3. Sharpening
# '''
# # Invert colors
# # imgInvert = 255 - img

# # Gamma correction
# img_gamma = img/255.0
# im_power_law_transformation = cv2.pow(img_gamma,2)

# # Sharpening
# kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
# mauImage = cv2.filter2D(src=im_power_law_transformation, ddepth=-1, kernel=kernel)


# # ----------------------------------------------- javi --------------------------------------------- #
# # '''
# #     1. Median Blur
# #     2. Registration
# #     3. Adaptive Gaussian Thresholding
# # '''
# # img2 = cv2.imread(PATH_TO_OUTPUT+'image32/fragment27.jpg', 0)

# # # Median Blur
# # blur = cv2.medianBlur(img2,1)

# # # Registration
# # c = 255/(np.log(1 + np.max(blur))) 
# # log_transformed = c * np.log(1 + blur) 
# # log_transformed = np.array(log_transformed, dtype = np.uint8)

# # # Adaptive Gaussian Thresholding
# # javiImage = cv2.adaptiveThreshold(log_transformed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
# #             cv2.THRESH_BINARY,11,2)


# # ----------------------------------------------- wil ---------------------------------------------- #
# # '''
# #     1. Fragmentation
# #     2. Gray scale
# #     3. Histogram
# # '''
# # # Gray scale conversion
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # Histogram
# # wilImage = cv2.equalizeHist(gray)


# # ----------------------------------------------- Image visualization --------------------------------------------- #
# cv2.imshow('normalImage', img)
# cv2.imshow('mau', mauImage)
# # cv2.imshow('wil', wilImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 