# import cv2
# from sklearn.cluster import KMeans
# import numpy as np
# from matplotlib import pyplot as plt
# import argparse

PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'

# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
# args = vars(ap.parse_args())

# src = cv2.imread(PATH_TO_OUTPUT+'image32/fragment28.jpg')
# # src = cv2.imread(PATH_TO_DATASET+'0193.TIF')

# # # Gray scale conversion
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# # Histogram
# src = cv2.equalizeHist(gray)


# # SIFT
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(src, None)
# kp_image = cv2.drawKeypoints(src, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# print(des1.shape)

# # FAST
# fast = cv2.FastFeatureDetector_create()
# kp2 = fast.detect(src, None)
# kp_image2 = cv2.drawKeypoints(src, kp2, None, color=(255,0,0))
# # kp_image2 = cv2.drawKeypoints(src, kp2, None, color=(255,0,0))
# print(len(kp2))

# # a = np.asarray(kp_image2,dtype=np.float32)/255
# # h, w, c = a.shape

# # # reshape the image to be a list of pixels
# # image = a.reshape((kp_image2.shape[0] * kp_image2.shape[1], 3))

# # # cluster the pixel intensities
# # clt = KMeans(n_clusters = args["clusters"])
# # clt.fit(image)

# # # build a histogram of clusters and then create a figure
# # # representing the number of pixels labeled to each color

# # centroids = clt.cluster_centers_
# # labels = clt.labels_

# # a2k = centroids[labels]
# # a3k = a2k.reshape(h, w, c)

# # kmeansImage = cv2.cvtColor(a3k, cv2.COLOR_RGB2BGR)


# cv2.imshow('SIFT', kp_image)
# cv2.imshow('FAST', kp_image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




from turtle import color
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from imports import filter_matches



# path of image 1
img1 = cv.imread(PATH_TO_OUTPUT+'image32/fragment28.jpg')

# # Gray scale conversion
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# path of image 2
path2 = "test.jpeg"

img2 = cv.imread(path2)

detector = cv.BRISK_create()
norm = cv.NORM_HAMMING

# finding features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)
result1 = cv.drawKeypoints(img1, kp1, None)
# cv.imwrite("features1.jpg", result1)
result2 = cv.drawKeypoints(img2, kp2, None)
# cv.imwrite("features2.jpg", result2)
# finding matches
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
matcher = cv.FlannBasedMatcher(flann_params, {})
raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
# finding good matches
p1, p2, kp_pairs, good = filter_matches(kp1, kp2, raw_matches)
if len(p1) >= 4:
    H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
    matchesNum = np.sum(status)
    totalMatches = len(status)
    print('%d / %d  inliers/matched' % (matchesNum, totalMatches))
    array = [matchesNum, totalMatches]
    labels = 'Number of Matches', 'Total Possible Matches'
    fig1, ax1 = plt.subplots()
    ax1.pie(array, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal') 
else:
    H, status = None, None
    print('%d matches found, not enough for homography estimation' % len(p1))
    fig = plt.figure(figsize=(5, 1.5))
    text = fig.text(0.5, 0.5, 'Matches found, not enough for homography estimation',
                ha='center', va='center')
    # text.set_path_effects([path_effects.Normal()])
# vis = explore_match("win", img1, img2, kp1, kp2, good, status, H)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None)

# cv2.imshow('SIFT', kp_image)
 # Equal aspect ratio ensures that pie is drawn as a circle.
cv.imshow('FAST', img3)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
