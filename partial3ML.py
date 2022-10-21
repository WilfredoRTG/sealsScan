import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
PATH_TO_DATASET = 'dataSet/'
PATH_TO_OUTPUT = 'maskedImages/'
PATH_TO_TRAIN = 'testImages/foca/test-enhanced.png'

# image = cv2.imread(PATH_TO_OUTPUT+'image32/fragment28.jpg', flags=cv2.IMREAD_COLOR)  
# for index, filename in enumerate(os.listdir(PATH_TO_DATASET)):
# image = cv2.imread(PATH_TO_DATASET + filename, flags=cv2.IMREAD_COLOR)
# image = cv2.imread(PATH_TO_DATASET + "DJI_0195.TIF", flags=cv2.IMREAD_COLOR)
image = cv2.imread(PATH_TO_TRAIN, flags=cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
# plt.imshow(segmented_image)
# plt.show()

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 0
masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
cv2.imwrite("test3.jpg", masked_image)
# cv2.imwrite(PATH_TO_OUTPUT+"image"+str(index+1)+".jpg", masked_image)
# plt.imshow(masked_image)
# plt.show()