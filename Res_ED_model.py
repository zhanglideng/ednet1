import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # 考虑一下bias的设置
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # inplace啥意思？
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv(x)
        output = self.relu(output)
        output = self.conv(output)
        output = output + x
        return output


class D_ResBlock(nn.Module):
    def __init__(self):
        super(D_ResBlock, self).__init__()
        self.res_block = ResBlock()

    def forward(self, x):
        output = self.res_block(x)
        output = self.res_block(output)
        output = self.res_block(output)
        output = output + x
        return output


class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(128, k + 1, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.d_res_block = D_ResBlock()
        self.deconv1 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, dilation=1, output_padding=1)

    def forward(self, x):
        # 当中有个3D卷积的步骤，考虑一下
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x1 = self.d_res_block(x)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = x1 + x
        x2 = self.conv3(x1)
        indices_map = torch.LongTensor([self.k]).cuda()
        indices_feature = torch.LongTensor([i for i in range(self.k)]).cuda()
        # attention map
        attention_map = torch.index_select(x2, 1, indices_map)
        x2 = torch.index_select(x2, 1, indices_feature)
        x2.mul(attention_map)
        # 解码器部分
        x2 = self.relu(self.deconv1(x2))
        x3 = self.d_res_block(x2)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = x3 + x2
        x3 = self.relu(self.deconv2(x3))
        x3 = self.relu(self.deconv3(x3))
        return x3
