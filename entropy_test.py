import os

import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# matplotlib.use('Qt5Agg')

entropies = []
points_per_side_array = []
points_per_side_array_human = []


def iterate_images():
    for filename in os.listdir(os.getcwd() + "/train/images")[:132]:
        # convert to grayscale
        img = cv2.imread(os.path.join(os.getcwd() + "/train/images", filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # calculate entropy
        entropy_img = entropy(img, disk(10)).mean()
        entropies.append(entropy_img)
        points_per_side = entropy_to_points_per_side(entropy_img)
        points_per_side_array.append(points_per_side)
        print(filename, entropy_img, entropy_to_points_per_side(entropy_img))
        # write to file
        write_number_to_file('label_images_entropy.txt', points_per_side, filename)


def write_number_to_file(filename, number, image_name):
    with open(filename, 'a') as f:
        f.write(image_name + "," + str(number) + "\n")


def entropy_to_points_per_side(entropy):
    if entropy < 2:
        return 4
    if entropy > 7:
        return 28

    return round(translate(entropy, 2, 7, 4, 28))


def plot_label_images():
    global points_per_side_array_human
    data = pd.read_csv("label_images.txt", header=None, names=["image", "pps"])
    points_per_side_array_human = data.get('pps')
    print(data)
    plt.boxplot(data.get('pps'))
    plt.show()
    plt.savefig("label_images.png")


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def plot_entropies():
    # plot a boxplot
    print(entropies)
    plt.boxplot(entropies)
    plt.show()
    plt.boxplot(points_per_side_array)
    plt.show()
    plt.savefig("entropy.png")


def print_correlations():
    data = pd.read_csv("label_images.txt", header=None, names=["image", "pps"])
    X = data.get('pps')
    Y = entropies
    pearson = X.corr(pd.Series(Y))
    spearman = X.corr(pd.Series(Y), method='spearman')
    print("Pearson correlation:", pearson)
    print("Spearman correlation:", spearman)

if __name__ == '__main__':
    plot_label_images()
    iterate_images()
    plot_entropies()
    print_correlations()

    #plot all arrays in one boxplot
    X = points_per_side_array_human
    Y = entropies
    Z = points_per_side_array
    data = pd.DataFrame({'human annotation': X, 'image entropies': Y, 'translated image entropies': Z})
    sns.boxplot(data=data)
    plt.show()
    plt.savefig("boxplots.png")

