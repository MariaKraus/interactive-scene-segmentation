import argparse
import os

import cv2
from sklearn.utils import shuffle
from interactive_learning import cnn
from sklearn.model_selection import train_test_split


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
    #images = load_images(image_dir)
    #numbers = read_numbers_from_file(label)

    #print("images", len(images))
    #print("numbers", len(numbers))
    interactive_trainer = cnn.CNNTrainer()

    #for i in range(len(images)):
        # image_resized = cv2.resize(images[i], (200, 200))
        #interactive_trainer.update(image_resized, int(numbers[i]))

    names = []
    numbers = []
    # Read the text file line by line
    with open(os.getcwd() + "/label_hamburg.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                names.append(parts[0])
                numbers.append(int(parts[1]))

    # Read the text file line by line
    with open(os.getcwd() + "/label_cats.txt", 'r') as file:
        lines  = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                names.append(parts[0])
                numbers.append(int(parts[1]))

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(names, numbers, test_size=0.2, random_state=42)

    epochs = 25
    for ep in range(epochs):
        print("Epoch: ", ep)
        train_data, train_labels = shuffle(train_data, train_labels)
        for i in range(len(train_data)):
            img = cv2.imread(os.path.join(os.getcwd(), "train", "cats", train_data[i]))
            image_resized = cv2.resize(img, (100, 100))
            interactive_trainer.update(image_resized, train_labels[i])

        # validate at the end of each epoch
        for k in range(len(val_data)):
            img = cv2.imread(os.path.join(os.getcwd(), "train", "cats", val_data[k]))
            image_resized = cv2.resize(img, (100, 100))
            interactive_trainer.validate(image_resized, val_labels[k])

    interactive_trainer.plot_results()
    # interactive_trainer.save_model()


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