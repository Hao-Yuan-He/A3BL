
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import os

class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            self.base_model = resnet50(num_classes=1000, weights=ResNet50_Weights.IMAGENET1K_V2)
            # Replace the last layer with a new one
            self.base_model.fc = nn.Linear(2048, num_classes)
        else:
            self.base_model = resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28), weight: str = None):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, 3), nn.ReLU())

        feature_map_size = (np.array(image_size) // 2 - 2) // 2 - 2
        num_features = 16 * feature_map_size[0] * feature_map_size[1]

        self.fc1 = nn.Sequential(nn.Linear(num_features, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)
        if weight and os.path.exists(weight): 
            self.load_weights(weight)
            # freeze all layers except the last one
            # for name, param in self.named_parameters():
            #     if 'fc' not in name:
            #         param.requires_grad = False
        
    def feature_map(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


class SymbolNet(nn.Module):
    def __init__(self, num_classes=4, image_size=(28, 28, 1)):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(32, momentum=0.99, eps=0.001),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64, momentum=0.99, eps=0.001),)

        num_features = 64 * (image_size[0] // 4 - 1) * (image_size[1] // 4 - 1)
        self.fc1 = nn.Sequential(nn.Linear(num_features, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(84, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class SymbolNetAutoencoder(nn.Module):
    def __init__(self, num_classes=4, image_size=(28, 28, 1)):
        super(SymbolNetAutoencoder, self).__init__()
        self.base_model = SymbolNet(num_classes, image_size)
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Sequential(nn.Linear(num_classes, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, image_size[0] * image_size[1]), nn.ReLU())

    def forward(self, x):
        x = self.base_model(x)
        # x = self.softmax(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
