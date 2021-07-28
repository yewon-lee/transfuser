from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

class ContrastiveLearningModel(nn.Module):
    def __init__(self, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        self.ImageCNN = models.resnet34(pretrained=True)
        self.ImageCNN.fc = nn.Sequential()
        self.normalize = normalize
        self.LidarEncoder = models.resnet18()
        self.LidarEncoder.fc = nn.Sequential()
        _tmp = self.LidarEncoder.conv1
        self.LidarEncoder.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
                                            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding,
                                            bias=_tmp.bias)
        self.flatten = nn.Sequential(
            nn.Flatten()
        )

    def forward(self, image, lidar):
        if self.normalize:
            image = normalize_imagenet(image)
        image_ft = self.ImageCNN(image)
        lidar_ft = self.LidarEncoder(lidar)
        return image_ft, lidar_ft # dims: 512

class ImitationLearningModel(nn.Module):
    def __init__ (self, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        self.ImageCNN = models.resnet34(pretrained=True)
        self.ImageCNN.fc = nn.Sequential()
        self.normalize = normalize
        self.LidarEncoder = models.resnet18()
        self.LidarEncoder.fc = nn.Sequential()
        _tmp = self.LidarEncoder.conv1
        self.LidarEncoder.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        self.flatten = nn.Sequential(
                    nn.Flatten()
                )
        self.fullyconn = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(100, 10),
            nn.ReLU(True),
            nn.Linear(10, 1)
        )
        embed_dim = num_classes
        self.merge = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, image, lidar):
        if self.normalize:
            image = normalize_imagenet(image)
        image_ft = self.ImageCNN(image)
        lidar_ft = self.LidarEncoder(lidar)
        final_ft = (image_ft,lidar_ft)
        final_ft = torch.cat(final_ft, dim=1)
        final_ft = self.merge(final_ft)
        final_ft = self.flatten(final_ft)
        final_ft = self.fullyconn(final_ft)
        return final_ft


class ImitationLearningModel_ImageOnly(nn.Module):
    def __init__ (self, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        self.ImageCNN = models.resnet34(pretrained=True)
        self.ImageCNN.fc = nn.Sequential()
        self.normalize = normalize
        self.flatten = nn.Sequential(
                    nn.Flatten()
                )
        self.fullyconn = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 10),
            nn.ReLU(True),
            nn.Linear(10, 1)
        )
    def forward(self, image):
        if self.normalize:
            image = normalize_imagenet(image)
        image_ft = self.ImageCNN(image)
        final_ft = image_ft
        final_ft = self.flatten(final_ft)
        final_ft = self.fullyconn(final_ft)
        return final_ft

class ImitationLearningModel_LidarOnly(nn.Module):
    def __init__ (self, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        self.LidarEncoder = models.resnet18()
        self.LidarEncoder.fc = nn.Sequential()
        _tmp = self.LidarEncoder.conv1
        self.LidarEncoder.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        self.flatten = nn.Sequential(
                    nn.Flatten()
                )
        self.fullyconn = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 10),
            nn.ReLU(True),
            nn.Linear(10, 1)
        )
    def forward(self, lidar):
        lidar_ft = self.LidarEncoder(lidar)
        final_ft = lidar_ft
        final_ft = self.flatten(final_ft)
        final_ft = self.fullyconn(final_ft)
        return final_ft

class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        inputs = normalize_imagenet(inputs)
        return self.features(inputs)




class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18()
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = self._model(lidar_data)
        return features

class ImitationLearningModel_Contrastive(nn.Module):
    def __init__ (self, contrastive_model, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        #self.ImageCNN = models.resnet34(pretrained=True)
        #self.ImageCNN.fc = nn.Sequential()
        self.normalize = normalize
        #self.LidarEncoder = models.resnet18()
        #self.LidarEncoder.fc = nn.Sequential()
        #_tmp = self.LidarEncoder.conv1
        #self.LidarEncoder.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
        #    kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        self.contrastive = contrastive_model
        self.flatten = nn.Sequential(
                    nn.Flatten()
                )
        self.fullyconn = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            #nn.Dropout(p=0.2),
            nn.Linear(100, 10),
            nn.ReLU(True),
            nn.Linear(10, 1)
        )
        embed_dim = num_classes
        self.merge = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, image, lidar):
        if self.normalize:
            image = normalize_imagenet(image)
        with torch.no_grad():
        	image_ft, lidar_ft = self.contrastive(image, lidar)
        final_ft = (image_ft,lidar_ft)
        final_ft = torch.cat(final_ft, dim=1)
        final_ft = self.merge(final_ft)
        final_ft = self.flatten(final_ft)
        final_ft = self.fullyconn(final_ft)
        return final_ft

class ControlsModel_FC(nn.Module):
    def __init__ (self, contrastive_model, num_classes=512, in_channels=2, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.contrastive = contrastive_model
        self.flatten = nn.Sequential(
                    nn.Flatten()
                )
        self.fullyconn = nn.Sequential(
            nn.Linear(512, 1),
        )
        embed_dim = num_classes
        self.merge = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, image, lidar):
        if self.normalize:
            image = normalize_imagenet(image)
        with torch.no_grad():
            image_ft, lidar_ft = self.contrastive(image, lidar)
        final_ft = (image_ft,lidar_ft)
        final_ft = torch.cat(final_ft, dim=1)
        final_ft = self.merge(final_ft)
        final_ft = self.flatten(final_ft)
        final_ft = self.fullyconn(final_ft)        
        return final_ft

