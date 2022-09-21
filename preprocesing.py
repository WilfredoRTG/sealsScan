from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
import argparse
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(PATH_TO_OUTPUT+'image1/fragment30.jpg', flags=cv2.IMREAD_COLOR)
cv2.imshow("image", image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
colors = utils.plot_colors(hist, clt.cluster_centers_, hexa=True)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Load the Summer Palace photo
# china = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])

# image = load_sample_image("fragment30.jpg")
# image = np.array(image, dtype=np.float64) / 255

# # Load Image and transform to a 2D numpy array.
# w, h, d = original_shape = tuple(image.shape)
# assert d == 3
# image_array = np.reshape(image, (w * h, d))

# print("Fitting model on a small sub-sample of the data")
# image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
# # 
# clt = KMeans(n_clusters=args["clusters"], random_state=0).fit(image_array_sample)
# # 

# # Get labels for all points
# print("Predicting color indices on the full image (k-means)")
# labels = clt.predict(image_array)


# codebook_random = shuffle(image_array, random_state=0, n_samples=args["clusters"])
# print("Predicting color indices on the full image (random)")
# labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)


# def recreate_image(codebook, labels, w, h):
#     """Recreate the (compressed) image from the code book & labels"""
#     return codebook[labels].reshape(w, h, -1)


# plt.figure(2)
# plt.clf()
# plt.axis("off")
# plt.title("K-Means")
# plt.imshow(recreate_image(clt.cluster_centers_, labels, w, h))