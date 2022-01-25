import torch.nn as nn
from torchvision import models


def create_efficientnet_b1(num_class: int, pretrained: bool = False):
    # Ideal input size: (240, 240)
    model = models.efficientnet_b1(pretrained)
    model.classifier[1] = nn.Linear(1280, num_class)
    return model


def create_efficientnet_b2(num_class: int, pretrained: bool = False):
    # Ideal input size: (260, 260)
    model = models.efficientnet_b2(pretrained)
    model.classifier[1] = nn.Linear(1408, num_class)
    return model


def create_resnet18(num_class: int, pretrained: bool = False):
    # Ideal input size: (224, 224)
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_class)
    return model


if __name__ == "__main__":
    from utils import model_summary

    # net = create_resnet18(num_class=2)
    # model_summary(net)

    net = create_efficientnet_b1(num_class=2)
    model_summary(net)

