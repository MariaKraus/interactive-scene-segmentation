import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt, cm
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchvision.models import VGG16_Weights


class CNNPretrained(nn.Module):
    # Defining the CNN architecture

    def __init__(self):
        super(CNNPretrained, self).__init__()
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        in_features = self.model._modules['classifier'][-1].in_features
        out_features = 1
        self.model._modules['classifier'][-1] = nn.Linear(in_features, out_features, bias=True)
        print(self.model._modules['features'])
        print(self.model._modules['classifier'])

    def forward(self, x):
        return self.model.forward(x)


class ImageDataset(data.Dataset):

    def __init__(self, transform=None):
        self.data = []
        # normalize the images to imagenet mean and std
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=None),  # Transformation for model that was trained on ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Transformation for model that was trained on ImageNet
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_image(self, image, delta):
        self.data.append((self.transform(image), np.float32(delta)))


class CNNTrainer:

    def __init__(self):
        self.batches = 0
        self.validation_mse = []
        self.validation_mae = []
        self.model = CNNPretrained()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.losses = []
        # Initialize TensorBoard writer
        self.writer = SummaryWriter()
        self.visualize_iteration = 50

    def train_one_batch(self, trainloader):
        running_loss = 0.0
        last_loss = 0.0
        mean_absolute_error = 0.0
        total_distance_to_label = 0.0
        visualize = self.batches % self.visualize_iteration == 0

        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch

            # Enable gradient calculation for the input image
            if visualize:
                inputs.requires_grad = True

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            mean_absolute_error += self.mae(outputs.squeeze(), labels.squeeze()).item()
            total_distance_to_label += torch.abs(outputs.squeeze() - labels.squeeze())
            loss.backward()

            # Visualize gradients
            if visualize:
                gradients = inputs.grad
                # Plot the magnitude of gradients as a heatmap
                gradient_magnitude = gradients.norm(dim=1, keepdim=True)  # Calculate gradient magnitude
                heatmap = gradient_magnitude[0]
                heatmap = np.maximum(heatmap, 0)
                # normalize the heatmap
                heatmap /= torch.max(heatmap)
                # draw the heatmap
                heatmap = heatmap.cpu().detach().numpy()
                heatmap = heatmap.squeeze()
                # revert the normalization
                img = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                )(inputs)
                img = img.squeeze().cpu().detach().numpy()
                # transform to rgb image
                img = np.uint8(255 * img)
                # convert the heatmap to RGB
                heatmap = np.uint8(255 * heatmap)

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # combine the heatmap with the original image
                superimposed_img = heatmap * 0.6 + img.transpose(1, 2, 0)
                # save the image to disk
                cv2.imwrite(f'result_images_pretrained/{self.batches}_map.jpg', superimposed_img)
                cv2.imwrite(f'result_images_pretrained/{self.batches}_img.jpg', img.transpose(1, 2, 0))

            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        avg_absolute_error = mean_absolute_error / len(trainloader)

        # Log loss to TensorBoard
        self.writer.add_scalar('MSE Loss/ train', avg_loss, self.batches)
        self.writer.add_scalar('MAE Loss/ train', avg_absolute_error, self.batches)


        return avg_loss

    def update(self, image, delta):
        image = np.nan_to_num(image, nan=0.0)
        image[np.isinf(image)] = 0.0
        self.model.train()
        # train one batch at a time
        self.batches += 1
        dataset = ImageDataset()
        dataset.add_image(image, delta)
        trainloader = data.DataLoader(dataset, batch_size=1, num_workers=1)
        loss = self.train_one_batch(trainloader)
        self.losses.append(loss)

    def validate(self, image, delta):
        image = np.nan_to_num(image, nan=0.0)
        image[np.isinf(image)] = 0.0
        self.model.eval()
        val_loss = 0.0
        mean_absolute_error = 0.0
        total_distance_to_label = 0.0
        validation_set = ImageDataset()
        validation_set.add_image(image, delta)
        validation_loader = data.DataLoader(validation_set, batch_size=1, num_workers=1)
        with torch.no_grad():
            for i, val_data in enumerate(validation_loader, 0):
                inputs, labels = val_data
                inputs[torch.isnan(inputs)] = 0
                inputs[torch.isinf(inputs)] = 0
                outputs = self.model(inputs)
                outputs[torch.isnan(outputs)] = 0
                outputs[torch.isinf(outputs)] = 0
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                val_loss += loss.item()
                mean_absolute_error += self.mae(outputs.squeeze(), labels.squeeze()).item()

        self.validation_mse.append(val_loss / len(validation_loader))
        self.validation_mae.append(mean_absolute_error / len(validation_loader))

    def log_validation(self):
        print("Logs validation")
        self.writer.add_scalar('Avg MSE Loss/ validation', sum(self.validation_mse) / len(self.validation_mse), self.batches)
        self.writer.add_scalar('Avg MAE Loss/ validation', sum(self.validation_mae) / len(self.validation_mae), self.batches)
        self.validation_mse = []
        self.validation_mae = []

    def plot_results(self):
        # Plot loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('batches')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

    def save_model(self):
        path = os.getcwd() + '/models'
        if not os.path.exists(path):
            os.makedirs(path)
        # save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = 'models/interactive_model_pretrained{}.pth'.format(timestamp, self.batches)
        torch.save(self.model.state_dict(), model_path)
