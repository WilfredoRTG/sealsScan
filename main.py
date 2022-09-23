import preprocesing
import histogram
import argparse
import matplotlib.pyplot as plt
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
ap.add_argument("-i", "--image", required = True, type = int, help = "# of image")
ap.add_argument("-f", "--fragment", required = True, type = int, help = "# of fragment")
args = vars(ap.parse_args())

def main():
    start_time = time.time()
    preprocesing.preprocesing(args)
    histogram.histogram(args)
    end_time = time.time()
    
    print("Time of execution: ", end_time - start_time)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()