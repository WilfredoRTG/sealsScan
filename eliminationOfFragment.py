from histogram import histogramGeneration
from numpy import load
import os
from timSort import timSort
import shutil

PATH_TO_CROP = 'cropImages/'
umbral = load('umbral.npy')

# for image in os.listdir(PATH_TO_CROP):
#     for imageFragment in os.listdir(PATH_TO_CROP + image):
#         repetitions = {}
#         colorsFrag = histogramGeneration(PATH_TO_CROP + image + "/" + imageFragment, kmeans=True)[0]
#         # print(colorsFrag)
#         for i in range(len(colorsFrag)):
#             if colorsFrag[i] in umbral:
#                 print(imageFragment, "eliminar")
#                 src_path = PATH_TO_CROP + image + "/" + imageFragment 
#                 dst_path = "eliminatedImages/" + image + imageFragment
#                 shutil.move(src_path, dst_path)
#     break

image = "image29/"

for imageFragment in os.listdir(PATH_TO_CROP + image):
    repetitions = {}
    colorsFrag = histogramGeneration(PATH_TO_CROP + image + "/" + imageFragment, kmeans=True)[0]
    # print(colorsFrag)
    for i in range(len(colorsFrag)):
        if colorsFrag[i] in umbral:
            print(imageFragment, "eliminar")
            src_path = PATH_TO_CROP + image + "/" + imageFragment 
            dst_path = "eliminatedImages/" + "image31" + imageFragment
            shutil.move(src_path, dst_path)