import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--restart", required = True, type = int, help = "true or false")
args = vars(ap.parse_args())

PATH_TO_TEST = "testImages/"

for type in os.listdir(PATH_TO_TEST):
    for index, filename in enumerate(os.listdir(PATH_TO_TEST+type)):
        if args["restart"] == 1:
            index += 100
        elif args["restart"] == 0:
            index += 1
        path = PATH_TO_TEST+type+"/"+filename
        pathOutput = PATH_TO_TEST+type+"/"+str(index)+".png"
        os.rename(path, pathOutput)