import torch
import torch.nn as nn
from model.unet_parts import MCB, ConvBNReLU, DSC


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_1 = ConvBNReLU(3, 64)
        self.conv1_2 = ConvBNReLU(64, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = ConvBNReLU(64, 128)
        self.conv2_2 = ConvBNReLU(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = MCB(128, 256)
        self.conv3_2 = MCB(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = MCB(256, 512)
        self.conv4_2 = MCB(512, 512)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = MCB(512, 512)
        self.conv5_2 = MCB(512, 512)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)

        pool1 = self.pool1(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)

        pool2 = self.pool2(conv2_2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)

        pool3 = self.pool3(conv3_2)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)

        pool4 = self.pool4(conv4_2)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        return conv1_2, conv2_2, conv3_2, conv4_2, conv5_2

def backbone():
    model = Backbone()
    return model
