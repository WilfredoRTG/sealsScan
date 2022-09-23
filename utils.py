# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hex(r, g, b):
    return ('#{:X}{:X}{:X}').format(r, g, b)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids, hexa=False):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    # loop over the percentage of each cluster and the color of
    # each cluster
    colors = []
    percentages = []
    RGBColors = []
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        colorRGB = (color*255).astype("uint8").tolist()
        RGBColors.append(color.astype("uint8").tolist())
        colors.append(rgb_to_hex(colorRGB[0],colorRGB[1], colorRGB[2]))
        # colors.append(colorRGB)
        percentages.append(percent)
    ax = plt.axes()
    ax.set_facecolor("black")
    plt.xlabel("Hexadecimal Colors")
    plt.ylabel("Percentage")
    plt.title("Percentage of each color")

    for i in range(len(colors)):
        plt.bar(colors[i], percentages[i], 
        color = [colors[i]], 
        width = 0.4)
    plt.show()

    if hexa:
        return colors
    else:
        return RGBColors