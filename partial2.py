import cv2 
import numpy as np
PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'


trainImg = cv2.imread('test-enhanced.png')
trainImg2 = cv2.imread('test2-enhanced.png')
# trainImg3 = cv2.imread('test4_1.png')

newIm = np.concatenate((trainImg, trainImg2), axis=0)


gray2 = cv2.cvtColor(newIm, cv2.COLOR_BGR2GRAY)
# trainGray = cv2.equalizeHist(gray2)
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
trainGray = cv2.filter2D(src=gray2, ddepth=-1, kernel=kernel)


img = cv2.imread(PATH_TO_OUTPUT+'image32/fragment28.jpg', flags=cv2.IMREAD_COLOR)
# img = cv2.imread(PATH_TO_DATASET+'0195.TIF', flags=cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram
wilImage = cv2.equalizeHist(gray)


# Sharpening
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
wilImage = cv2.filter2D(src=wilImage, ddepth=-1, kernel=kernel)


orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(trainGray, None)
kp2, des2 = orb.detectAndCompute(wilImage, None)

matcher = cv2.BFMatcher()
matches = matcher.match(des1, des2)


finalImage = cv2.drawMatches(wilImage, kp2, trainGray, kp1, matches[:20], None)

# cv2.imshow('finalImage', newIm)
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resized_Window", 499, 374)
cv2.imshow('Resized_Window', finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()