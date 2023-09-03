import argparse
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from interactive_learning import cnn
from interactive_learning import cnn_pretrained
from interactive_learning import cnn_classification
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


def apply_custom_transform(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomGrayscale(),
        transforms.GaussianBlur(3),
    ])

    transformed_image = transform(image)
    return transformed_image


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


def train(image_dir: str, label: str, model: int, augmentations: int, epochs : int):
    """
    Train the model
    :param epochs:
    :param augmentations: The number of augmentations to apply to each image
    :param model: The model to use
    :param image_dir: The directory with the training images
    :param label: The directory with the training labels
    :return: None
    """
    interactive_trainer = None
    if model == 1:
        interactive_trainer = cnn.CNNTrainer()
    elif model == 2:
        interactive_trainer = cnn_pretrained.CNNTrainer()
    elif model == 3:
        interactive_trainer = cnn_classification.CNNTrainer()

    names = []
    numbers = []
    # Read the text file line by line
    with open(label, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                names.append(parts[0])
                numbers.append(int(parts[1]))

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    # Split the data into training and validation sets
    train_names, val_names, train_numbers, val_numbers = train_test_split(names, numbers, test_size=0.2, random_state=42)

    # Load the training images and augment them
    for i in tqdm(range(len(train_names)), desc="Loading and Augmenting Images", unit="image"):
        img = cv2.imread(os.path.join(image_dir, train_names[i]))
        # resize otherwise it takes too long
        img = cv2.resize(img, (224, 224))
        train_data.append(img)
        train_labels.append(train_numbers[i])
        for j in range(augmentations):
            img_augmented = apply_custom_transform(img)
            train_data.append(img_augmented)
            train_labels.append(train_numbers[i])

    # Load the validation images
    for n in tqdm(range(len(val_names)), desc="Loading and Augmenting Images", unit="image"):
        img = cv2.imread(os.path.join(image_dir, val_names[n]))
        # resize otherwise it takes too long
        img = cv2.resize(img, (224, 224))
        val_data.append(img)
        val_labels.append(val_numbers[n])


    # Train the model for a fixed numer of epochs
    epochs = epochs
    for ep in range(epochs):
        print("\nEpoch:", ep)
        train_data, train_labels = shuffle(train_data, train_labels)
        for i in tqdm(range(len(train_data)), desc="training", unit="image"):
            interactive_trainer.update(train_data[i], train_labels[i])

        val_data, val_labels = shuffle(val_data, val_labels)
        # validate at the end of each epoch
        for k in tqdm(range(len(val_data)), desc="validation", unit="image"):
            interactive_trainer.validate(val_data[k], val_labels[k])

    interactive_trainer.plot_results()
    interactive_trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Scene Segmentation")
    parser.add_argument("--i_dir", type=str, default=os.getcwd() + "/train/random/random_images/",
                        help="The directory with the training images")
    parser.add_argument("--label", type=str, default=os.getcwd() + "/train/random/label.txt",
                        help="The directory with the training labels")
    parser.add_argument("--model", type=int, default=3,  # 1 = cnn, 2 = vgg16, 3 = classifier
                        help="The model to use for training")
    parser.add_argument("--augmentations", type=int, default=15,
                        help="The number of augmentations to use for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="The number of augmentations to use for training")

    args = parser.parse_args()

    if not os.path.isdir(args.i_dir):
        print("The given directory: ", args.dir, " does not exist.")
        exit(0)

    train(image_dir=args.i_dir, label=args.label, model=args.model, augmentations=args.augmentations, epochs=args.epochs)
