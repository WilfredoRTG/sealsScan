import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt


PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'

# img = cv2.imread(PATH_TO_OUTPUT+'image32/fragment28.jpg', flags=cv2.IMREAD_COLOR)
MIN_MATCH_COUNT = 10
img1 = cv.imread('test-enhanced.png',0)          # queryImage
img3 = cv.imread('test2-enhanced.png',0)          # queryImage
img2 = cv.imread(PATH_TO_OUTPUT+'image16/fragment26.jpg', flags=cv.IMREAD_COLOR)

newIm = np.concatenate((img1, img3), axis=0)


gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#-------------------------------------------------
#Histogram
#wilImage = cv2.equalizeHist(gray)
#-------------------------------------------------

# Sharpening
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
wilImage = cv2.filter2D(src=img2, ddepth=-1, kernel=kernel)

#--------------------------------------------------------

#--------------------------------------------------------
# img2 = cv.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(newIm,None)
kp2, des2 = sift.detectAndCompute(wilImage,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = newIm.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    wilImage = cv.polylines(wilImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(newIm,kp1,wilImage,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()