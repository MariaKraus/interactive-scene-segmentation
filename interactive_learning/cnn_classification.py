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
        self.data.append((self.transform(image), np.float32(delta)))


class ProximityLoss(nn.Module):
    """
    Custom loss function that combines cross-entropy loss and proximity loss
    """

    def __init__(self, num_classes, proximity_weight=0.3, proximity_threshold=2):
        super(ProximityLoss, self).__init__()
        self.num_classes = num_classes
        self.proximity_weight = proximity_weight
        self.proximity_threshold = proximity_threshold

    def forward(self, predicted_logits, true_labels):
        # Calculate the cross-entropy loss
        cross_entropy_loss = nn.CrossEntropyLoss()(predicted_logits.squeeze(), true_labels)
        # Calculate the proximity loss based on the difference between predicted and true classes
        class_diff = torch.abs(predicted_logits.argmax(dim=1) - true_labels)
        # if the difference is less than the threshold, multiply by the weight, otherwise leave as is
        proximity_loss = torch.where(class_diff <= self.proximity_threshold, class_diff * self.proximity_weight,
                                     class_diff)
        # Combine cross-entropy loss and proximity loss
        combined_loss = cross_entropy_loss + proximity_loss.mean()
        return combined_loss


class CNNTrainer:

    def __init__(self):
        self.batches = 0
        self.model = CNNClassification()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Change to cross entropy loss
        self.criterion = ProximityLoss(num_classes=18, proximity_weight=0.5, proximity_threshold=2)
        self.cross_entropy_criterion = nn.CrossEntropyLoss()
        self.losses = []
        # Initialize TensorBoard writer
        self.writer = SummaryWriter()
        self.visualize_iteration = 50

    def train_one_batch(self, trainloader):
        running_loss = 0.0
        correct_predictions = 0  # Counter for correct predictions
        total_predictions = 0  # Counter for total predictions
        total_distance_to_label = 0
        total_cross_entropy_loss = 0
        visualize = self.batches % self.visualize_iteration == 0

        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.squeeze().long())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            total_cross_entropy_loss = self.cross_entropy_criterion(outputs.squeeze(), labels.squeeze().long()).item()
            # Calculate accuracy
            predicted_labels = outputs.argmax(dim=1)  # Get the predicted class labels
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            # Calculate the distance between the predicted and true labels
            total_distance_to_label += torch.abs(predicted_labels - labels.squeeze())

            # Visualize gradients
            if visualize:
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
                cv2.imwrite(f'result_images_classification/{self.batches}_map.jpg', superimposed_img)
                cv2.imwrite(f'result_images_classification/{self.batches}_img.jpg', img.transpose(1, 2, 0))

                # return into train mode
                self.model.train()
                self.model.set_cam(False)
                # GRAD CAM END #

        avg_custom_loss = running_loss / len(trainloader)
        accuracy = correct_predictions / total_predictions  # Calculate accuracy
        avg_distance_to_label = total_distance_to_label / len(trainloader)  # Calculate average distance to label
        avg_cross_entropy_loss = total_cross_entropy_loss / len(trainloader)  # Calculate average cross entropy loss

        # Log loss to TensorBoard
        self.writer.add_scalar('Cross Entropy Loss/ train', avg_cross_entropy_loss, self.batches)
        self.writer.add_scalar('Custom Entropy Loss/ train', avg_custom_loss, self.batches)
        self.writer.add_scalar('Accuracy/ train', accuracy, self.batches)
        self.writer.add_scalar('Distance/ train', avg_distance_to_label * 2, self.batches)

        return avg_custom_loss

    def update(self, image, delta):
        # train one batch at a time
        self.model.train()
        delta = delta / 2
        self.batches += 1
        dataset = ImageDataset()
        dataset.add_image(image, delta)
        trainloader = data.DataLoader(dataset, batch_size=1, num_workers=1)
        loss = self.train_one_batch(trainloader)
        self.losses.append(loss)

    def validate(self, image, delta):
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

        with torch.no_grad():
            for i, val_data in enumerate(validation_loader, 0):
                inputs, labels = val_data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.squeeze().long())
                val_loss += loss.item()
                cross_entropy_loss += self.cross_entropy_criterion(outputs.squeeze(), labels.squeeze().long())
                # Calculate accuracy
                predicted_labels = outputs.argmax(dim=1)  # Get the predicted class labels
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
                # Calculate the distance between the predicted and true labels
                total_distance_to_label = torch.abs(predicted_labels - labels.squeeze())

        accuracy = correct_predictions / total_predictions  # Calculate accuracy
        avg_distance_to_label = total_distance_to_label / len(validation_loader)  # Calculate average distance to label
        avg_cross_entropy_loss = cross_entropy_loss / len(validation_loader)
        avg_custom_loss = val_loss / len(validation_loader)

        self.writer.add_scalar('Cross Entropy Loss/ validation', avg_cross_entropy_loss, self.batches)
        self.writer.add_scalar('Custom Entropy Loss/ validation', avg_custom_loss, self.batches)
        self.writer.add_scalar('Accuracy/ validation', accuracy, self.batches)
        self.writer.add_scalar('Distance/ validation', avg_distance_to_label * 2, self.batches)

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
