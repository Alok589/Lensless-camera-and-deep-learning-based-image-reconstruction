from skimage import transform
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import skimage.io
import numpy as np
import scipy.io as sio
import torch
from torch.optim import optimizer
from torch.nn.modules import BatchNorm2d


class residual_block(nn.Module):
    """(conv1 + relu => conv2 + relu => BN)"""

    def __init__(self, in_ch, out_ch):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = (nn.ReLU(inplace=True),)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class Res_Unet(nn.Module):
    def __init__(self, in_ch=1):
        super(Res_Unet, self).__init__()
        self.conv1 = conv_bn(in_ch, 32)
        self.residual_1 = residual_block(32, 32)

        self.conv2 = conv_bn(32, 64)
        self.residual_2 = residual_block(64, 64)

        self.conv3 = conv_bn(64, 128)
        self.residual_3 = residual_block(128, 128)

        self.conv4 = conv_bn(128, 256)
        self.residual_4 = residual_block(256, 256)

        # have to make convTranspose class
        #


if __name__ == "__main__":
    model = Res_Unet(1)
    input_image = torch.rand(size=(1, 32, 32))
    out = model(input_image)
    print(out.shape)