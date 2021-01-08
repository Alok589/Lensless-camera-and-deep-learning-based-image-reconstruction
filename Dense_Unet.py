from skimage import transform
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import skimage.io
import numpy as np
import scipy.io as sio
import torch
from torch.optim import optimizer

tform = transform.SimilarityTransform(rotation=0.00174)


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            # nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv2(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(double_conv2(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class upnocat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upnocat, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


# class Dense_Unet(nn.Module):
#     def __init__(self, PhiL=None, PhiR=None, n_channels=4):
#         super(Dense_Unet, self).__init__()
#         self.inc = inconv(n_channels, 128)
#         self.down1 = down(128, 256)
#         self.down2 = down(256, 512)
#         self.down3 = down(512, 1024)
#         self.down4 = down(1024, 1024)
#         self.up1 = up(2048, 512)
#         self.up2 = up(1024, 256)
#         self.up3 = up(512, 128)
#         self.up4 = up(256, 128)
#         self.outc = outconv(128, 3)
#         if np.all(PhiL):
#             self.PhiL = nn.Parameter(torch.tensor(PhiL))
#         else:
#             self.PhiL = nn.Parameter(torch.randn(500, 256, 1))
#         if np.all(PhiR):
#             self.PhiR = nn.Parameter(torch.tensor(PhiR))
#         else:
#             self.PhiR = nn.Parameter(torch.randn(620, 256, 1))
#         self.bn = nn.BatchNorm2d(4, momentum=0.99)
#         # self.drop = nn.Dropout(p=0.3)


class Dense_Unet(nn.Module):
    def __init__(self, n_channels=4):
        super(Dense_Unet, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 3)
        self.PhiL = nn.Parameter(torch.randn(500, 256, 1))
        self.PhiR = nn.Parameter(torch.randn(620, 256, 1))
        self.bn = nn.BatchNorm2d(4, momentum=0.99)

    def forward(self, Xinp):

        X0 = F.leaky_relu(
            torch.matmul(
                torch.matmul(Xinp[:, 0, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                self.PhiL[:, :, 0],
            )
            .permute(0, 2, 1)
            .unsqueeze(3)
        )
        X11 = F.leaky_relu(
            torch.matmul(
                torch.matmul(Xinp[:, 1, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                self.PhiL[:, :, 0],
            )
            .permute(0, 2, 1)
            .unsqueeze(3)
        )
        X12 = F.leaky_relu(
            torch.matmul(
                torch.matmul(Xinp[:, 2, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                self.PhiL[:, :, 0],
            )
            .permute(0, 2, 1)
            .unsqueeze(3)
        )
        X2 = F.leaky_relu(
            torch.matmul(
                torch.matmul(Xinp[:, 3, :, :], self.PhiR[:, :, 0]).permute(0, 2, 1),
                self.PhiL[:, :, 0],
            )
            .permute(0, 2, 1)
            .unsqueeze(3)
        )
        Xout = torch.cat((X2, X12, X11, X0), 3)
        x = Xout.permute(0, 3, 1, 2)
        x = self.bn(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x
        # return torch.sigmoid(x), Xout


if __name__ == "__main__":
    model = Dense_Unet()
    input_image = torch.rand(size=(1, 4, 500, 620))
    out = model(input_image)
    print(out.shape)
