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


class CNN(nn.Module):
    # Defining the CNN architecture

    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for regression
        self.fc_layers = nn.Sequential(
            # 25 * 25 is the number of pixels in the image after 2 pooling layers
            nn.Linear(in_features=16 * int(220/4) * int(220/4), out_features=4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )

        self.fc_layers_output = nn.Sequential(
            nn.Linear(in_features=4, out_features=1),
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
            transforms.ToTensor(),
            transforms.Resize((220, 220), antialias=None),  # Transformation for model that was trained on ImageNet
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
        self.model = CNN()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.losses = []
        # Initialize TensorBoard writer
        self.writer = SummaryWriter()
        self.visualize_iteration = 500

    def train_one_batch(self, trainloader):
        running_loss = 0.0
        last_loss = 0.0
        mean_absolute_error = 0.0
        total_distance_to_label = 0.0
        visualize = self.batches % self.visualize_iteration == 0

        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            inputs[torch.isnan(inputs)] = 0
            inputs[torch.isinf(inputs)] = 0

            # Enable gradient calculation for the input image
            if visualize:
                inputs.requires_grad = True

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs[torch.isnan(outputs)] = 0
            outputs[torch.isinf(outputs)] = 0
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
                heatmap[torch.isnan(heatmap)] = 0
                heatmap[torch.isinf(heatmap)] = 0
                # draw the heatmap
                heatmap = heatmap.cpu().detach().numpy()
                heatmap = heatmap.squeeze()
                # revert the normalization
                img = inputs
                img = img.squeeze().cpu().detach().numpy()
                # transform to rgb image
                img = np.uint8(255 * img)
                # convert the heatmap to RGB
                try:
                    # The code that generates the warning
                    heatmap = np.uint8(255 * heatmap)
                except RuntimeWarning as rw:
                    # Handle the warning here
                    print("Caught a RuntimeWarning:", rw)

                    # Get the specific value that triggered the warning
                    value_causing_warning = rw.args[0]
                    print("Value causing the warning:", value_causing_warning)

                    # Optionally, you can set the problematic values to a specific valid value
                    heatmap[np.isnan(heatmap)] = 0
                    heatmap[np.isinf(heatmap)] = 0
                else:
                    # Continue with the rest of your code
                    pass

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # combine the heatmap with the original image
                superimposed_img = heatmap * 0.6 + img.transpose(1, 2, 0)
                # save the image to disk
                cv2.imwrite(f'result_images/{self.batches}_map.jpg', superimposed_img)
                cv2.imwrite(f'result_images/{self.batches}_img.jpg', img.transpose(1, 2, 0))
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            running_loss += loss.item()
            last_loss = loss.item()

            # Analyze feature maps for a specific input after training
            if visualize:  # To analyze feature maps every 10 batches, adjust as needed
                # self.analyze_feature_maps(inputs[0])
                pass

        avg_loss = running_loss / len(trainloader)
        avg_absolute_error = mean_absolute_error / len(trainloader)
        avg_distance_to_label = total_distance_to_label / len(trainloader)

        # Log loss to TensorBoard
        self.writer.add_scalar('MSE Loss/ train', avg_loss, self.batches)
        self.writer.add_scalar('MAE Loss/ train', avg_absolute_error, self.batches)
        return avg_loss

    def update(self, image, delta):
        image = np.nan_to_num(image, nan=0.0)
        image[np.isinf(image)] = 0.0
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
        val_loss = 0.0
        mean_absolute_error = 0.0
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

        self.validation_mse.append(val_loss / len(validation_set))
        self.validation_mae.append(mean_absolute_error / len(validation_set))

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
        model_path = 'models/interactive_model_{}.pth'.format(timestamp, self.batches)
        torch.save(self.model.state_dict(), model_path)

    def analyze_feature_maps(self, inputs):
        # Set the model to evaluation mode
        self.model.eval()
        feature_maps = []

        # Choose a convolutional layer for analysis (e.g., the first convolutional layer)
        target_layer = self.model.conv_layers[0]
        second_layer = self.model.conv_layers[1]
        third_layer = self.model.conv_layers[2]
        fourth_layer = self.model.conv_layers[3]
        fifth_layer = self.model.conv_layers[4]

        # Forward pass to get feature maps
        with torch.no_grad():
            feature_maps.append(target_layer(inputs))
            feature_maps.append(second_layer(feature_maps[0]))
            feature_maps.append(third_layer(feature_maps[1]))
            feature_maps.append(fourth_layer(feature_maps[2]))
            feature_maps.append(fifth_layer(feature_maps[3]))

        # Convert the feature maps that should be depicted to numpy arrays
        feature_maps0 = feature_maps[0].squeeze().cpu().numpy()
        feature_maps3 = feature_maps[3].squeeze().cpu().numpy()

        # Plot the feature maps
        num_channels = feature_maps0.shape[0]
        num_cols = min(8, num_channels)  # Number of columns in the visualization
        num_rows = (num_channels + num_cols - 1) // num_cols
        fig0, axes0 = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        fig0.suptitle('Feature Map  0', fontsize=16)
        for i, ax in enumerate(axes0.flatten()):
            if i < num_channels:
                ax.imshow(feature_maps0[i], cmap='viridis')
                ax.set_title(f'Channel {i}')
            ax.axis('off')
        plt.savefig(f'result_images/{self.batches}_featureMaps0.png')
        plt.clf()

        # Plot the feature maps
        num_channels = feature_maps3.shape[0]
        num_cols = min(8, num_channels)  # Number of columns in the visualization
        num_rows = (num_channels + num_cols - 1) // num_cols
        fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        fig3.suptitle('Feature Map 3', fontsize=16)
        for i, ax in enumerate(axes3.flatten()):
            if i < num_channels:
                ax.imshow(feature_maps3[i], cmap='viridis')
                ax.set_title(f'Channel {i}')
            ax.axis('off')

        plt.tight_layout()
        # Save the feature map
        plt.savefig(f'result_images/{self.batches}_featureMaps3.png')
        plt.clf()
        self.model.train()
