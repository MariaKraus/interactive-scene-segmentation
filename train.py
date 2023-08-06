import argparse
import os

import cv2
from sklearn.utils import shuffle
from interactive_learning import cnn


def load_images(directory: str):
    # Read image from user input
    directory_path = ""
    images = []
    # directory exists
    while not images:
        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    return images

def read_numbers_from_file(filename):
    numbers = []
    with open(filename, 'r') as file:
        for line in file:
            number = int(line.strip())
            numbers.append(number)
    return numbers

def train(image_dir: str, label:str):
    """
    Train the model
    :param image_dir: The directory with the training images
    :param label: The directory with the training labels
    :return: None
    """
    images = load_images(image_dir)
    numbers = read_numbers_from_file(label)

    print("images", len(images))
    print("numbers", len(numbers))
    interactive_trainer = cnn.CNNTrainer()

    for i in range(len(images)):
        image_resized = cv2.resize(images[i], (100, 100))
        #interactive_trainer.update(image_resized, int(numbers[i]))

    hamburg_names = []
    hamburg_numbers = []
    # Read the text file line by line
    with open(os.getcwd() + "/label_hamburg.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                hamburg_names.append(parts[0])
                hamburg_numbers.append(int(parts[1]))

    epochs = 100

    for ep in range(epochs):
        print("Epoch: ", ep)
        hamburg_names, hamburg_numbers = shuffle(hamburg_names, hamburg_numbers)
        for i in range(len(hamburg_names)):
            img = cv2.imread(os.path.join(os.getcwd(), "train", "hamburg", hamburg_names[i]))
            image_resized = cv2.resize(img, (100, 100))
            interactive_trainer.update(image_resized, hamburg_numbers[i])

    interactive_trainer.plot_results()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Scene Segmentation")
    parser.add_argument("--i_dir", type=str, default=os.getcwd() + "/train/ulm/",
                        help="The directory with the training images")
    parser.add_argument("--label", type=str, default=os.getcwd() + "/label.txt",
                        help="The directory with the training labels")

    args = parser.parse_args()

    if not os.path.isdir(args.i_dir):
        print("The given directory: ", args.dir, " does not exist.")
        exit(0)

    train(image_dir=args.i_dir, label=args.label)