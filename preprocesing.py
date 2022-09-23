from sklearn.cluster import KMeans
import utils
import cv2
import numpy as np
import argparse
import time 

PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
ap.add_argument("-i", "--image", required = True, type = str, help = "# of clusters")
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
start = time.time()
image = cv2.imread(PATH_TO_DATASET+args["image"]+'.TIF')
cv2.namedWindow("Normal Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Normal Image", 998, 748)
cv2.imshow("Normal Image", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

a = np.asarray(image,dtype=np.float32)/255
h, w, c = a.shape

# reshape the image to be a list of pixels
image = a.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color

centroids = clt.cluster_centers_
labels = clt.labels_

a2k = centroids[labels]
a3k = a2k.reshape(h, w, c)

kmeansImage = cv2.cvtColor(a3k, cv2.COLOR_RGB2BGR)
cv2.namedWindow("K-means Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("K-means Image", 998, 748)

endTime = time.time()
print("Time of execution: ", endTime - start)
cv2.imshow("K-means Image", kmeansImage)
hist = utils.centroid_histogram(clt)
colors = utils.plot_colors(hist, clt.cluster_centers_, hexa=True)


cv2.waitKey(0)
cv2.destroyAllWindows()
