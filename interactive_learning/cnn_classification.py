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

class CNNClassification(nn.Module):
    # Defining the CNN architecture

    def __init__(self):
        super(CNNClassification, self).__init__()
        # get the pretrained VGG16 network
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # disect the network to access its last convolutional layer
        self.features_conv = self.model.features[:30]

        # get the max pool of the features stem
        self.max_pool = self.model.features[30]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg16
        self.classifier = self.model.classifier
        # replace the last layer of the classifier with a fully connected layer with 18 classes
        self.classifier[-1] = nn.Linear(in_features=4096, out_features=18, bias=True)
        # placeholder for the gradients
        self.gradients = None
        self.cam = False

        # print the model
        print("model features: ", self.model._modules['features'])
        print("model classifier: ", self.model._modules['classifier'])

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        """
        Activation hook for the gradients, needed for the heatmap
        :param grad: the gradients
        :return: None
        """
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if self.cam:
            h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        # average the channels like in the vgg16 model
        x = self.model.avgpool(x)
        # flatten the vector
        x = torch.flatten(x, 1)
        # apply the classifier
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    def set_cam(self, cam):
        self.cam = cam


class ImageDataset(data.Dataset):
    """
    Dataset for the images
    """

    def __init__(self, transform=None):
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Normalization for model that was trained on ImageNet
            transforms.Resize((224, 224), antialias=None),  # Transformation for model that was trained on ImageNet
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_image(self, image, delta):
        """
        Add an image to the dataset
        :param image: the image
        :param delta: the label
        :return: none
        """
        self.data.append((self.transform(image), np.float32(delta)))

class CNNTrainer:
    """
    Class for training the CNN
    """
    def __init__(self):
        self.batches = 0
        self.validation_ce = []
        self.validation_acc = []
        self.validation_mae = []
        self.model = CNNClassification()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Change to cross entropy loss
        self.criterion = nn.CrossEntropyLoss()
        self.losses = []
        # Initialize TensorBoard writer
        self.writer = SummaryWriter()
        self.visualize_iteration = 50

    def train_one_batch(self, trainloader):
        """
        Train one batch of the dataset (batch size = 1)
        :param trainloader: load image
        :return: none
        """
        running_loss = 0.0
        correct_predictions = 0  # Counter for correct predictions
        total_predictions = 1  # Counter for total predictions
        total_distance_to_label = 0
        avg_custom_loss = 1
        visualize = self.batches % self.visualize_iteration == 0

        # Train one batch
        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            inputs[torch.isnan(inputs)] = 0
            inputs[torch.isinf(inputs)] = 0
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs[torch.isnan(outputs)] = 0
            outputs[torch.isinf(outputs)] = 0
            loss = self.criterion(outputs.squeeze(), labels.squeeze().long())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            # Calculate accuracy
            predicted_labels = outputs.argmax(dim=1)  # Get the predicted class labels
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            # Calculate the distance between the predicted and true labels
            total_distance_to_label += torch.abs(predicted_labels - labels.squeeze())

            # Visualize gradients
            if visualize:
                try:
                    # Check if the folder already exists
                    if not os.path.exists(os.path.join(os.getcwd() + "/heatmap_classification/")):
                        # If it doesn't exist, create the folder
                        os.makedirs(os.path.join(os.getcwd() + "/heatmap_classification/"))
                    # GRAD CAM #
                    self.model.eval()
                    self.model.set_cam(True)
                    pred = self.model(inputs)
                    # get the gradient of the output with respect to the parameters of the model ?????????????
                    pred[:, int(labels[0].item())].backward()
                    # pull the gradients out of the model
                    gradients = self.model.get_activations_gradient()
                    # pool the gradients across the channels
                    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # get the activations of the last convolutional layer
                    activations = self.model.get_activations(inputs).detach()
                    # weight the channels by corresponding gradients
                    for i in range(512):
                        activations[:, i, :, :] *= pooled_gradients[i]
                    # average the channels of the activations
                    heatmap = torch.mean(activations, dim=1).squeeze()
                    heatmap[torch.isnan(heatmap)] = 0
                    heatmap[torch.isinf(heatmap)] = 0
                    # relu on top of the heatmap
                    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
                    heatmap = np.maximum(heatmap, 0)
                    # normalize the heatmap
                    heatmap /= torch.max(heatmap)
                    # draw the heatmap
                    heatmap = heatmap.cpu().detach().numpy()

                    # revert the normalization
                    img = transforms.Normalize(
                        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                    )(inputs)

                    img = img.squeeze().cpu().detach().numpy()

                    # transform to rgb image
                    img = np.uint8(255 * img)
                    # rezize the heatmap to have the same size as the image
                    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[2]))
                    # convert the heatmap to RGB

                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    # combine the heatmap with the original image
                    superimposed_img = heatmap * 0.4 + img.transpose(1, 2, 0)
                    # save the image to disk
                    cv2.imwrite(f'heatmap_classification/{self.batches}_map.jpg', superimposed_img)
                except RuntimeWarning as w:
                    print(w)
                    print("Error in heatmap creation")
                cv2.imwrite(f'heatmap_classification/{self.batches}_img.jpg', img.transpose(1, 2, 0))

                # return into train mode
                self.model.train()
                self.model.set_cam(False)
                # GRAD CAM END #

            accuracy = correct_predictions / total_predictions  # Calculate accuracy
            avg_distance_to_label = total_distance_to_label / len(trainloader)  # Calculate average distance to label
            avg_cross_entropy_loss = running_loss / len(trainloader)  # Calculate average cross entropy loss

            # Log loss to TensorBoard
            self.writer.add_scalar('Cross Entropy Loss/ train', avg_cross_entropy_loss, self.batches)
            self.writer.add_scalar('Accuracy/ train', accuracy, self.batches)
            self.writer.add_scalar('MAE Loss/ train', avg_distance_to_label * 2, self.batches)

        return avg_custom_loss

    def update(self, image, delta):
        """
        Update the model with one batch
        :param image: the image
        :param delta: the label
        :return: none
        """
        image = np.nan_to_num(image, nan=0.0)
        image[np.isinf(image)] = 0.0
        # train one batch at a time
        self.model.train()
        delta = delta / 2
        self.batches += 1
        dataset = ImageDataset()
        dataset.add_image(image, delta)
        # intialize the dataloader
        trainloader = data.DataLoader(dataset, batch_size=1, num_workers=1)
        loss = self.train_one_batch(trainloader)
        # collect all losses
        self.losses.append(loss)

    def validate(self, image, delta):
        """
        Validate the model
        :param image: the image
        :param delta: the label
        :return: none
        """
        image = np.nan_to_num(image, nan=0.0)
        image[np.isinf(image)] = 0.0
        self.model.eval()
        delta = delta / 2
        val_loss = 0.0
        cross_entropy_loss = 0.0
        correct_predictions = 0  # Counter for correct predictions
        total_predictions = 0  # Counter for total predictions
        total_distance_to_label = 0
        validation_set = ImageDataset()
        validation_set.add_image(image, delta)
        validation_loader = data.DataLoader(validation_set, batch_size=1, num_workers=1)

        # no gradient calculation
        with torch.no_grad():
            for i, val_data in enumerate(validation_loader, 0):
                inputs, labels = val_data
                inputs[torch.isnan(inputs)] = 0
                inputs[torch.isinf(inputs)] = 0
                outputs = self.model(inputs)
                outputs[torch.isnan(outputs)] = 0
                outputs[torch.isinf(outputs)] = 0
                loss = self.criterion(outputs.squeeze(), labels.squeeze().long())
                val_loss += loss.item()
                # Calculate accuracy
                predicted_labels = outputs.argmax(dim=1)  # Get the predicted class labels
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
                # Calculate the distance between the predicted and true labels
                total_distance_to_label = torch.abs(predicted_labels - labels.squeeze())

        # collect all losses
        self.validation_acc.append(correct_predictions / total_predictions)  # Calculate accuracy
        self.validation_mae.append(total_distance_to_label * 2 / len(validation_loader))  # Calculate average distance to label
        self.validation_ce.append(val_loss / len(validation_loader))

    def log_validation(self):
        """
        Log the validation results to TensorBoard
        :return: none
        """
        print("Logs validation")
        self.writer.add_scalar('Cross Entropy Loss/ validation', sum(self.validation_ce)/len(self.validation_ce), self.batches)
        self.writer.add_scalar('Accuracy/ validation', sum(self.validation_acc)/len(self.validation_acc), self.batches)
        self.writer.add_scalar('Avg MAE Loss/ validation', sum(self.validation_mae)/len(self.validation_mae), self.batches)

    def plot_results(self):
        """
        Plot all the losses
        :return: none
        """
        # Plot loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('batches')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

    def save_model(self):
        """
        Save the model
        :return: none
        """
        path = os.getcwd() + '/models'
        if not os.path.exists(path):
            os.makedirs(path)
        # save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = 'models/interactive_model_classification{}.pth'.format(timestamp, self.batches)
        torch.save(self.model.state_dict(), model_path)
