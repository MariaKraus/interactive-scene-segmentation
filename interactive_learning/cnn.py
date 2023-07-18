from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


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
            nn.Linear(in_features=128 * 4 * 4, out_features=1024),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x


class ImageDataset(data.Dataset):
    def __init__(self, images, deltas, transform=None):
        self.images = images
        self.deltas = deltas
        self.data = dict(zip(images, deltas))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.data[index]


class CNNTrainer:

    def __init__(self):
        self.trainloader = None
        self.model = CNN()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch_inex, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch_inex + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        return last_loss

    def update(self, images, deltas, epoch_index):
        # train one epoch at a time
        # since we do incremental learning, the caller has to keep track of epoch count
        dataset = ImageDataset(images, deltas)
        self.trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        loss = self.train_one_epoch(epoch_index, writer)

        # TODO: potentially think of a way to add a validation set (if that even makes sense)

        writer.add_scalar('Loss/train', loss, epoch_index)
        writer.flush()

        model_path = 'models/fashion_{}.pth'.format(timestamp, epoch_index)
        torch.save(self.model.state_dict(), model_path)
