import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt, cm
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    # Defining the CNN architecture

    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for regression
        self.fc_layers = nn.Sequential(
            # 25 * 25 is the number of pixels in the image after 2 pooling layers
            nn.Linear(in_features=32 * 50 * 50, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.fc_layers_output = nn.Sequential(
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        # Forward pass through layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.fc_layers_output(x)
        return x


class ImageDataset(data.Dataset):

    def __init__(self, transform=None):
        self.data = []  # TODO: change to dict where key is path to image and value is delta
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomGrayscale(),
            transforms.GaussianBlur(3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_image(self, image, delta):
        self.data.append((transforms.ToTensor()(image), np.float32(delta)))

    def augment_image(self, image, delta):
        self.data.append((self.transform(image), np.float32(delta)))


class CNNTrainer:

    def __init__(self):
        self.batches = 0
        self.model = CNN()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.losses = []
        # Initialize TensorBoard writer
        self.writer = SummaryWriter()

    def train_one_batch(self, trainloader):
        running_loss = 0.0
        last_loss = 0.0
        mean_absolute_error = 0.0

        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            mean_absolute_error += self.mae(outputs.squeeze(), labels.squeeze()).item()
            loss.backward()
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()
            # Log validation loss to TensorBoard
            self.writer.add_scalar('output/ validation', outputs.squeeze()[0], self.batches)

        avg_loss = running_loss / len(trainloader)
        avg_absolute_error = mean_absolute_error / len(trainloader)

        # Log loss to TensorBoard
        self.writer.add_scalar('MSE Loss/ train', avg_loss, self.batches)
        self.writer.add_scalar('Avg Absolute Error/ train', avg_absolute_error, self.batches)

        return avg_loss

    def update(self, image, delta):
        # train one batch at a time
        self.batches += 1
        dataset = ImageDataset()
        dataset.add_image(image, delta)
        # augment a fixed number of times
        augmentations = 9
        for i in range(augmentations):
            dataset.augment_image(image, delta)

        trainloader = data.DataLoader(dataset, batch_size=augmentations + 1, num_workers=1)
        loss = self.train_one_batch(trainloader)
        self.losses.append(loss)

    def validate(self, image, delta):
        val_loss = 0.0
        mean_absolute_error = 0.0
        validation_set = ImageDataset()
        validation_set.add_image(image, delta)
        validation_loader = data.DataLoader(validation_set, batch_size=1, num_workers=1)

        for i, val_data in enumerate(validation_loader, 0):
            inputs, labels = val_data
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            val_loss += loss.item()
            mean_absolute_error += self.mae(outputs.squeeze(), labels.squeeze()).item()

        self.writer.add_scalar('MSE Loss/ validation', val_loss / len(validation_set), self.batches)
        self.writer.add_scalar('Avg Absolute Error/ valdiation', mean_absolute_error / len(validation_set),
                               self.batches)

    def plot_results(self):
        # Plot loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('batches')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

    def save_model(self):
        path = os.getcwd() + '/models'
        if not os.path.exists(path):
            os.makedirs(path)
        # save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = 'models/interactive_model_{}.pth'.format(timestamp, self.batches)
        torch.save(self.model.state_dict(), model_path)
