import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.conv6_1 = nn.Conv2d(1024, 2048, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(2048, 2048, 3, 1, 1)
        self.conv6_3 = nn.Conv2d(2048, 2048, 3, 1, 1)

        self.deconv5 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn3 = nn.BatchNorm2d(3)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.bn2048 = nn.BatchNorm2d(2048)

    def forward(self, x):
        # INPUT:640*480*3 = 921,600
        # OUTPUT:320*240*64 = 4,915,200
        conv1 = self.bn64(self.relu(self.conv1_1(x)))
        conv1 = self.bn64(self.relu(self.conv1_2(conv1)))
        conv1 = self.maxpool(conv1)

        # INPUT:320*240*64 = 4,915,200
        # OUTPUT:160*120*128 = 2,457,600
        conv2 = self.bn128(self.relu(self.conv2_1(conv1)))
        conv2 = self.bn128(self.relu(self.conv2_2(conv2)))
        conv2 = self.maxpool(conv2)

        # INPUT:160*120*128 = 2,457,600
        # OUTPUT:80*60*256 = 1,228,800
        conv3 = self.bn256(self.relu(self.conv3_1(conv2)))
        conv3 = self.bn256(self.relu(self.conv3_2(conv3)))
        conv3 = self.bn256(self.relu(self.conv3_3(conv3)))
        conv3 = self.maxpool(conv3)

        # INPUT:80*60*256 = 1,228,800
        # OUTPUT:40*30*512 = 614,400
        conv4 = self.bn512(self.relu(self.conv4_1(conv3)))
        conv4 = self.bn512(self.relu(self.conv4_2(conv4)))
        conv4 = self.bn512(self.relu(self.conv4_3(conv4)))
        conv4 = self.maxpool(conv4)

        # INPUT:40*30*512 = 614,400
        # OUTPUT:80*60*256 = 1,228,800
        deconv3 = self.bn256(self.relu(self.deconv3(conv4)))

        # INPUT:80*60*256 = 1,228,800
        # OUTPUT:160*120*128= 2,457,600
        deconv2 = self.bn128(self.relu(self.deconv2(deconv3)))

        # INPUT:160*120*128 = 2,457,600
        # OUTPUT:320*240*64 = 4,915,200
        deconv1 = self.bn64(self.relu(self.deconv1(deconv2)))

        # INPUT:320*240*64 = 4,915,200
        # OUTPUT:640*480*3 = 921,600
        output = self.bn3(self.relu(self.deconv(deconv1)))

        # OUTPUT:640*480*3 = 921,600
        return output
