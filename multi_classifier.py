import os
from style_classifier.model import StyleClassifier
from style_classifier.utils import random_data_generator, target_data_generator, get_confusion_matrix
from torchvision import transforms
import torch
from PIL import Image
import gradio
import numpy as np
import gradio.preprocessing_utils
import gradio.inputs
import gradio.outputs


def test(pre_trained_path, labels, sample_per_label):
    data = []
    for item in labels:
        data.extend(target_data_generator(item, sample_per_label))

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = StyleClassifier()

    model.load_state_dict(torch.load(pre_trained_path, map_location='cpu')['model_state_dict'])

    pred = []

    for item in data:
        im = Image.open(os.path.join('../style_data_clean', item))
        im = im.convert('RGB')

        predict = model(data_transforms(im).unsqueeze(0)).tolist()[0]

        max_index = 0
        temp_max = predict[0]
        for i in range(1, len(predict)):
            if predict[i] > temp_max:
                temp_max = predict[i]
                max_index = i

        pred.append(max_index)

    data_dict = {}
    for i, item in enumerate(labels):
        data_dict[item] = i

    for i, _ in enumerate(data):
        data[i] = data_dict[data[i].split('/')[0]]

    get_confusion_matrix(pred, data, labels).show()


def pre(inp):
    im = gradio.preprocessing_utils.decode_base64_to_image(inp)
    im = im.convert('RGB')
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    im = data_transforms_val(im).unsqueeze(0)

    return im.numpy()


def web_demo(model):
    inp = gradio.inputs.ImageUpload(preprocessing_fn=pre)
    out = gradio.outputs.Label(label_names=['lively', 'modern', 'pop_art', 'vintage'], num_top_classes=4)
    io = gradio.Interface(inputs=inp, outputs=out, model_type="pytorch", model=model)
    io.launch()


if __name__ == '__main__':
    # test('./190731_multi.pth.tar', ['lively', 'modern', 'pop_art', 'vintage'], 50)
    model = StyleClassifier(need_softmax=True)

    model.load_state_dict(torch.load('./190731_multi.pth.tar', map_location='cpu')['model_state_dict'])

    web_demo(model)
