import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=12, out_features=48),
            nn.ReLU(),
            nn.Linear(48, 16)
        )

        self.fc_layers2 = nn.Sequential(
            nn.Linear(in_features=24576, out_features=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(1, -1)
        x = self.fc_layers2(x)
        print(x)
        return x


class ImageDataset(data.Dataset):
    def __init__(self, transform=None):
        self.data = []  # TODO: change to dict where key is path to image and value is delta
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_image(self, image, delta):
        self.data.append((transforms.ToTensor()(image), np.float32(delta)))


class CNNTrainer:

    def __init__(self):
        self.epoch = 0
        self.model = CNN()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, trainloader):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()

            print('[%d, %5d] loss: %.3f' % (self.epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        return last_loss

    def update(self, image, delta):
        # train one epoch at a time
        # since we do incremental learning, the caller has to keep track of epoch count

        self.epoch += 1

        dataset = ImageDataset()
        dataset.add_image(image, delta)

        # summary(self.model.cuda(), (3, 100, 100))

        trainloader = data.DataLoader(dataset, batch_size=1, num_workers=1)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        loss = self.train_one_epoch(trainloader)

        # TODO: potentially think of a way to add a validation set (if that even makes sense)

        writer.add_scalar('Loss/train', loss, self.epoch)
        writer.flush()

        path = os.getcwd() + '/models'
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = 'models/interactive_model_{}.pth'.format(timestamp, self.epoch)
        torch.save(self.model.state_dict(), model_path)
