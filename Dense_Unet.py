from skimage import transform
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import skimage.io
import numpy as np
import scipy.io as sio
import torch
from torch.optim import optimizer

# tform = transform.SimilarityTransform(rotation=0.00174)


# class double_conv(nn.Module):
#     """(conv => BN => ReLU) * 2"""

#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding="same"),
#             nn.BatchNorm2d(out_ch, momentum=0.99),
#             # nn.SELU(inplace=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding="same"),
#             nn.BatchNorm2d(out_ch, momentum=0.99),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


class orange(nn.Module):
    """"""

    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super(orange, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding="same")
        self.relu = nn.ReLU(inplace=True)
        self.BatchNorm2d = nn.BatchNorm2d()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.relu(x)

        x = self.BatchNorm2d(x)
        return x


class Dense_Unet(nn.Module):
    """"""

    def __init__(self, in_ch, out_ch):
        super(model, self).__init__()
        self.orange1 = orange(1, 32)

    def forward(self, x):
        input_image = torch.rand(size=(1, 1, 128, 128))
        x1_in = nn.Conv2d(1, 32, 3, padding="same")(input_image)
        x1_in = nn.BatchNorm2d()(x1_in)
        x = self.orange1(x1_in)
        x1_out = torch.add([x, x1_in])

        x2_in = nn.Conv2D(32, 64, 2, stride=2, padding="same")(x1_out)
        x2_in = nn.relu(inplace=True)(x2_in)
        x2_in = nn.BatchNorm2d()(x2_in)
        x = self.orange(x2_in)
        x2_out = torch.add([x, x2_in])

        x3_in = nn.Conv2D(64, 128, 2, stride=2, padding="same")(x2_out)
        x3_in = nn.relu(inplace=True)(x3_in)
        x3_in = nn.BatchNorm2d()(x3_in)
        x = orange(x3_in)
        x3_out = torch.add([x, x3_in])

        x4_in = nn.Conv2D(128, 256, 2, stride=2, padding="same")(x3_out)
        x4_in = nn.relu(inplace=True)(x4_in)
        x4_in = nn.BatchNorm2d()(x4_in)
        x = orange(x4_in)
        x4_out = torch.add([x, x4_in])

        y3_in = nn.ConvTranspose2d(256, 128, 2, stride=2)(x4_out)
        y3_in = nn.relu(inplace=True)(y3_in)
        y3_in = nn.BatchNorm2d()(y3_in)
        cat1 = torch.cat([x3_out, y3_in])

        y3_in = nn.Conv2D(256, 128, 3, padding="same")(cat1)
        y3_in = nn.relu(inplace=True)(y3_in)
        y3_in = BatchNorm2d()(y3_in)
        x = orange(y3_in)
        y3_out = torch.add([x, y3_in])

        y2_in = nn.ConvTranspose2d(128, 64, 2, stride=2)(y3_out)
        y2_in = nn.relu(inplace=True)(y2_in)
        y2_in = nn.BatchNorm2d()(y2_in)
        cat2 = torch.cat([x2_out, y2_in])

        y2_in = nn.Conv2D(128, 64, 3, padding="same")(cat2)
        y2_in = nn.relu(inplace=True)(y2_in)
        y2_in = BatchNorm2d()(y2_in)
        x = orange(y2_in)
        y2_out = torch.add([x, y2_in])

        y1_in = nn.ConvTranspose2d(64, 32, 2, stride=2)(y2_out)
        y1_in = nn.relu(inplace=True)(y1_in)
        y1_in = nn.BatchNorm2d()(y1_in)
        cat3 = torch.cat([x1_out, y1_in])

        y1_in = nn.Conv2D(64, 32, 3, padding="same")(cat3)
        y1_in = nn.relu(inplace=True)(y1_in)
        y1_in = BatchNorm2d()(y1_in)
        x = orange(y1_in)
        y1_out = torch.add([x, y1_in])

        y_out = nn.Conv2D(32, 64, 5, padding="same")(y1_out)
        y_out = nn.BatchNorm2d()(y_out)
        y_out = nn.Conv2D(64, 1)(y1_out)

        return y1_out


if __name__ == "__main__":
    input_image = torch.rand(size=(8, 1, 128, 128))
    model = Dense_Unet(input_image, y1_out)
    # out = model(input_image, y1_out)
    print(out.shape)


# class residual_block(nn.Module):
#     """(conv => BN => ReLU) * 2"""

#     def __init__(self, in_ch, out_ch):
#         super(double_conv2, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch, momentum=0.99),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch, momentum=0.99),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(double_conv(in_ch, out_ch))

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=False):
#         super(up, self).__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 2)

#         self.conv = double_conv(in_ch, out_ch)

#     # def forward(self, x1, x2):
#     #     x1 = self.up(x1)
#     #     diffX = x1.size()[2] - x2.size()[2]
#     #     diffY = x1.size()[3] - x2.size()[3]
#     #     x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
#     #     x = torch.cat([x2, x1], dim=1)
#     #     x = self.conv(x)
#     #     return x

#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x


# class upnocat(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(upnocat, self).__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x


# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 7, padding=1)
#         # self.bn = nn.BatchNorm2d(out_ch, momentum= 0.99)

#     def forward(self, x):
#         x = self.conv(x)
#         # x = self.bn(x)
#         return x


# class Dense_Unet(nn.Module):
#     def __init__(self, n_channels=1):
#         super(Dense_Unet, self).__init__()
#         self.inc = inconv(n_channels, 32)

#         self.down1 = double_conv(1, 32)
#
#         self.down1 = down(32, 64)
#
#         self.down2 = down(64, 128)
#
#         self.down3 = down(128, 256)
#
#         # self.down4 = down(256, 256)
#         self.up1 = up(256, 128)
#
#         self.up2 = up(128, 64)
#
#         self.up3 = up(64, 32)
#
#         self.outc = outconv(64, 1)

#     def forward(self, x):

#         block1_op = self.block1(x)

#         return x


# class Dense_Unet(nn.Module):
#     def __init__(self, n_channels=1):
#         super(Dense_Unet, self).__init__()
#         self.inc = inconv(n_channels, 32)
#         self.res1 = residual_block(32, 32
#         self.down1 = down(32, 64)
#         self.res2 = residual_block(64, 64)
#         self.down2 = down(64, 128)
#         self.res2 = residual_block(128, 128)
#         self.down3 = down(128, 256)
#         self.res3 = residual_block(256, 256)
#         # self.down4 = down(256, 256)
#         self.up1 = up(256, 128)
#         self.res4 = residual_block(128, 128)
#         self.up2 = up(128, 64)
#         self.res5 = residual_block(64, 64)
#         self.up3 = up(64, 32)
#         self.res6 = residual_block(32, 32)
#         self.outc = outconv(64, 1)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # x5 = self.down4(x4)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         # x = self.up4(x, x1)
#         x = self.outc(x)

#         return x

# return torch.sigmoid(x), Xout
