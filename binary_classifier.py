import os
from style_classifier.model import StyleClassifier
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from style_classifier.utils import random_data_generator, target_data_generator
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(pre_train_model_path, target_tag, sample_num):
    model = StyleClassifier()
    model.load_state_dict(torch.load(pre_train_model_path, map_location='cpu')['model_state_dict'])

    print(torch.load(pre_train_model_path, map_location='cpu')['epoch'])

    data = random_data_generator(sample_num)
    # data = target_data_generator(target_tag, sample_num)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    plt.figure(figsize=(2, sample_num * 0.4), dpi=600)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=1.2)

    modern_acc = []
    other_acc = []
    for i, img in enumerate(data):
        im = Image.open(os.path.join('../style_data_clean', img))
        im = im.convert('RGB')
        plt.subplot(sample_num, 1, i + 1)
        plt.axis('off')
        plt.imshow(im)
        predict = model(data_transforms(im).unsqueeze(0)).item()
        if img.split('/')[0] == target_tag:
            if predict > 0.5:
                modern_acc.append(1)
            elif predict < 0.5:
                modern_acc.append(0)
        if img.split('/')[0] != target_tag:
            if predict < 0.5:
                other_acc.append(1)
            elif predict > 0.5:
                other_acc.append(0)
        plt.title('Label=%s, Possibility of %s=%s' % (img.split('/')[0], target_tag, str(predict)), fontsize=4)

    plt.savefig('./result.png', dpi=600)

    modern_acc = np.array(modern_acc)
    other_acc = np.array(other_acc)
    result = np.concatenate((modern_acc, other_acc))
    print('Total Accuracy: %s' % str(result.mean()))


if __name__ == '__main__':
    test('./190728_modern_gram.pth.tar', 'modern', 50)
