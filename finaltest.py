import numpy as np
import cv2 as cv
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

PATH_TO_DATASET = 'dataSet/DJI_'
PATH_TO_OUTPUT = 'cropImages/'
PATH_TO_OUTPUT_2 = 'cropImages2/'

MIN_MATCH_COUNT = 10
img0 = cv.imread('test0_enhanced.png',
                 flags=cv.IMREAD_COLOR)          # queryImage
img1 = cv.imread('test1_enhanced.jpg',
                 flags=cv.IMREAD_COLOR)          # queryImage
img2 = cv.imread('test2_enhanced.jpg', flags=cv.IMREAD_COLOR)
img3 = cv.imread('test3_enhanced.jpg', flags=cv.IMREAD_COLOR)
img4 = cv.imread('test4_enhanced.jpg', flags=cv.IMREAD_COLOR)
img5 = cv.imread('test5_enhanced.jpg', flags=cv.IMREAD_COLOR)

comparationImage = cv2.imread(
    PATH_TO_OUTPUT+'image16/fragment26.jpg', flags=cv2.IMREAD_COLOR)
# comparationImage = cv.imread(PATH_TO_OUTPUT_2+'image31/fragment289.jpg', flags=cv.IMREAD_COLOR)
# comparationImage = cv.imread('testImages/foca/32.jpg', flags=cv.IMREAD_COLOR)

newIm = np.concatenate((img0, img1), axis=0)
newIm2 = np.concatenate((img2, img3), axis=0)
newIm3 = np.concatenate((img4, img5), axis=0)
finalImage = np.concatenate((newIm, newIm2, newIm3), axis=1)


# Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
wilImage = cv2.filter2D(src=comparationImage, ddepth=-1, kernel=kernel)


# comparationImage = cv.imread('box_in_scene.png',0) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(finalImage, None)
kp2, des2 = sift.detectAndCompute(wilImage, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, c = finalImage.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    wilImage = cv.polylines(
        wilImage, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    print("Es foca")
else:
    print("Son piedras")
    # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
result = cv.drawMatches(finalImage, kp1, wilImage,
                        kp2, good, None, **draw_params)
# plt.imshow(img3, 'gray'),plt.show()

y_true = [
            "piedras", "piedras", "foca", "piedras", "foca", "piedras",
            "foca", "piedras", "foca", "piedras", "foca", "piedras", "piedras", 
            "foca", "foca", "piedras", "piedras", "piedras", "piedras", "foca", "foca", "foca"
        ]

y_pred = [
            "foca", "piedras", "foca", "piedras", "foca", "foca",
            "foca", "foca", "foca", "piedras", "foca", "foca", "foca", 
            "piedras", "foca", "piedras", "foca", "foca", "foca", "foca", "foca", "foca"
        ]

labels = ["foca", "piedras"]

score = accuracy_score(y_true, y_pred)
print("Accuracy: ", score)

cm = confusion_matrix(y_true, y_pred, labels=labels)

len_focas = []
len_rocas = []

for i in range(len(y_true)):
    if (y_true[i] == "foca"):
        len_focas.append(y_true[i])
    else:
        len_rocas.append(y_true[i])

accuracy_focas = cm[0][0]/(len(len_focas))
accuracy_rocas = cm[1][1]/(len(len_rocas))

print("Accuracy focas: ", accuracy_focas)

print("Accuracy rocas: ", accuracy_rocas)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot()
plt.show()

# cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

# # Using resizeWindow()
# cv2.resizeWindow("Resized_Window", 800, 604)

# # Displaying the image
# cv2.imshow("Resized_Window", result)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
