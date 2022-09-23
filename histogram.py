from timeit import repeat
from sklearn.cluster import KMeans
import argparse
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse

PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, type = int, help = "# of image")
ap.add_argument("-f", "--fragment", required = True, type = int, help = "# of fragment")
args = vars(ap.parse_args())


image = cv2.imread(PATH_TO_OUTPUT + 'image'+str(args['image'])+'/fragment'+str(args['fragment'])+'.jpg')
cv2.imshow("image", image)


colorsRep = {}
for i in range(len(image)):
    for j in range(len(image[i])):
        hexa = utils.rgb_to_hex(image[i][j][0], image[i][j][1], image[i][j][2])
        if hexa in colorsRep.keys():
            colorsRep[hexa] += 1
        else:
            colorsRep[hexa] = 1

l = list(colorsRep.values())
for i in range(len(l)):
    for j in range(i + 1, len(l)):

        if l[i] < l[j]:
            l[i], l[j] = l[j], l[i]

highers = l[:20]
colorsFrag = []
for key, value in colorsRep.items():
    for higher in highers:
        if value == higher:
            colorsFrag.append(key)
            break

ax = plt.axes()
ax.set_facecolor("black")
# plt.figure("Histogram in fragment")
plt.xlabel("Colors")
plt.ylabel("Repetitions")
plt.title("Predominant colors in fragment")

for i in range(len(colorsFrag)):
    plt.bar(colorsFrag[i], highers[i], 
            color = [colorsFrag[i]])

plt.show()