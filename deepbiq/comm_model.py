import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
from torchvision.models.alexnet import model_urls
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


def get_alexnet_pretrain_model():
    model_urls['alexnet'] = model_urls['alexnet'].replace('https://', 'http://')
    print("=> using pre-trained model '{}'".format('alexnet'))
    model = models.__dict__['alexnet'](pretrained=True)

    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, 5))
    new_classifier = torch.nn.Sequential(*mod)
    weight_init(list(new_classifier.children())[6])
    model.classifier = new_classifier

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    return model


def get_imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FeatureMode(object):
    def __init__(self, f):
        self.model = get_alexnet_pretrain_model()

        print("=> loading checkpoint '{}'".format(f))
        checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(f, checkpoint['epoch']))
        cudnn.benchmark = True

        mod = list(self.model.classifier.children())
        mod.pop()
        mod.pop()
        new_classifier = torch.nn.Sequential(*mod)
        self.model.classifier = new_classifier

        self.model.eval()

    def extract_feature(self, input):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = self.model(input_var)
        return output.data.cpu().numpy()


def get_crop_box(img_w, img_h, crop_w, crop_h, crop_num_w, crop_num_h):
    interval_w = (img_w - crop_w) / crop_num_w
    interval_h = (img_h - crop_h) / crop_num_h
    w = []
    h = []
    for i in range(crop_num_w):
        w.append(0 + i * interval_w)
    for i in range(crop_num_h):
        h.append(0 + i * interval_h)
    crop_box = []
    for i in h:
        for j in w:
            crop_box.append((j, i, j + crop_w, i + crop_h))
    return crop_box