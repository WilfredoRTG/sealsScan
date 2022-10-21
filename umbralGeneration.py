from histogram import histogramGeneration
import os 
from numpy import save

PATH_TO_WATER = 'testImages/agua/'
colorUmbral = {}
umbral = []


for image in os.listdir(PATH_TO_WATER):
    colorsFrag = histogramGeneration(PATH_TO_WATER + image, kmeans=True)[0]
    
    for i in range(len(colorsFrag)):
        if colorsFrag[i] in colorUmbral.keys():
            colorUmbral[colorsFrag[i]] += 1
        else:
            colorUmbral[colorsFrag[i]] = 1

    umbral = list(colorUmbral.keys())

print(umbral)
save('umbral.npy', umbral)