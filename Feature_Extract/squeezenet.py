from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
sqnet_feat_list = ['conv1', 'ReLU1', 'maxpool1',
                   'fire1',
                   'fire2',
                   'fire3', 'maxpool4',
                   'fire4',
                   'fire5',
                   'fire6',
                   'fire7', 'maxpool8',
                   'fire8',

                   ]
sqnet_classifier_list = ['Dropout9', 'conv10', 'ReLU11', 'avgpool12']


class SqueezeNet1_0(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(SqueezeNet1_0, self).__init__()
        # self.select_feats = ['maxpool1', 'maxpool4', 'maxpool8', 'fire8']
        # self.select_classifier = ['conv10']

        self.select_feats = ['maxpool1',
                             'fire3', 'maxpool4',
                             'fire4',
                             'fire5',
                             'fire6',
                             'fire7', 'maxpool8',
                             'fire8', ]
        self.select_classifier = ['Dropout9', 'conv10', 'ReLU11', 'avgpool12']

        self.feat_list = self.select_feats + self.select_classifier

        self.sqnet_feats = models.squeezenet1_0(pretrained=True).features
        self.sqnet_classifier = models.squeezenet1_0(
            pretrained=True).classifier
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        """Extract multiple feature maps."""
        features = []
        for name, layer in self.sqnet_feats._modules.items():
            x = layer(x)
            print(name)
            if sqnet_feat_list[int(name)] in self.feat_list:
                features.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        for name, layer in self.sqnet_classifier._modules.items():
            x = layer(x)
            print(name)
            if sqnet_classifier_list[int(name)] in self.feat_list:
                features.append(x)
        return features


# print(models.squeezenet1_0(pretrained=True))
# print(models.vgg11(pretrained=True))
# print(models.vgg19(pretrained=True))
