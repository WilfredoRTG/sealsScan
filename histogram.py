from sklearn.cluster import KMeans
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
import timSort as sort

def histogramGeneration(pathToImage, plot=False, kmeans=False):

    image = cv2.imread(pathToImage)
    if kmeans:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        a = np.asarray(image,dtype=np.float32)/255
        h, w, c = a.shape

        # reshape the image to be a list of pixels
        image = a.reshape((image.shape[0] * image.shape[1], 3))

        # cluster the pixel intensities
        clt = KMeans(n_clusters = 10)
        clt.fit(image)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color

        centroids = clt.cluster_centers_
        labels = clt.labels_

        a2k = centroids[labels]
        a3k = a2k.reshape(h, w, c)

        image = cv2.cvtColor(a3k, cv2.COLOR_RGB2BGR)*255
    
    # cv2.imshow("image", imageN)
    colorsRep = {}
    for i in range(len(image)):
        for j in range(len(image[i])):
            hexa = utils.rgb_to_hex(round(image[i][j][0]), round(image[i][j][1]), round(image[i][j][2]))
            if hexa in colorsRep.keys():
                colorsRep[hexa] += 1
            else:
                colorsRep[hexa] = 1

    l = list(colorsRep.values())
    sort.timSort(l)

    highers = l[-5:]

    colorsFrag = []
    for key, value in colorsRep.items():
        for higher in highers:
            if value == higher:
                colorsFrag.append(key)
                break

    colorsFrag.reverse()

    if plot:
        ax = plt.axes()
        ax.set_facecolor("white")
        # plt.figure("Histogram in fragment")
        plt.xlabel("Colors")
        plt.ylabel("Repetitions")
        plt.title("Predominant colors in fragment")

        for i in range(len(colorsFrag)):
            plt.bar(colorsFrag[i], highers[i], 
                    color = [colorsFrag[i]])

        plt.show()

    return colorsFrag, highers, colorsRep



