import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
from torchsummary import summary
ImageFile.LOAD_TRUNCATED_IMAGES = True


class StyleClassifier(nn.Module):
    def __init__(self):
        super(StyleClassifier, self).__init__()
        self.pre_trained = models.vgg16(pretrained=True)
        self.cnn = nn.Sequential(*list(self.pre_trained.children())[0])
        self.conv_dr = nn.Conv2d(512, 128, 1)
        self.fc = nn.Linear(in_features=128 * 128, out_features=1024)
        self.fc_1 = nn.Linear(in_features=1024, out_features=4)
        # self.sig = nn.Sigmoid()

    def get_style_gram(self, style_features):
        style_features = style_features.view(-1, 7 * 7, 128)
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
        h = self.conv_dr(h)

        h = h.permute(0, 2, 3, 1)
        h = self.get_style_gram(h)

        h = self.fc(h)

        h = self.fc_1(h)
        # h = F.log_softmax(h, dim=1)

        return h


if __name__ == '__main__':
    model = models.vgg16(pretrained=False)
    cnn = nn.Sequential(*list(model.children())[0])

    summary(cnn, (3, 224,224))
