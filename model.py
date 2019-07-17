import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


class StyleClassifier(nn.Module):
    def __init__(self):
        super(StyleClassifier, self).__init__()
        self.pre_trained = models.resnet34(pretrained=True)
        self.cnn = nn.Sequential(*list(self.pre_trained.children())[:-2])
        self.conv = nn.Conv2d(512, 128, 1, stride=1)
        self.fc = nn.Linear(in_features=128 * 128, out_features=1)
        self.sig = nn.Sigmoid()

    def get_style_gram(self, style_features):
        """
                get the gram matrix
                :param style_features:
                :return:
                """
        style_features = style_features.view(-1, 49, 128)
        style_features_t = style_features.permute(0, 2, 1)
        grams = []
        for i in range(len(style_features)):
            grams.append(torch.matmul(style_features_t[i], style_features[i]))

        grams = torch.stack(grams, 0)

        grams = grams.view(-1, 128 * 128)

        return grams

    def forward(self, x):
        h = x
        h = self.cnn(h)
        h = self.conv(h)

        h = h.permute(0, 2, 3, 1)
        h = self.get_style_gram(h)

        h = self.sig(self.fc(h))

        return h
