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


class orange(nn.Module):
    """"""

    def __init__(self, in_ch, out_ch):
        super(orange, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.BatchNorm2d = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.relu(x)

        x = self.BatchNorm2d(x)
        return x


class Dense_Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dense_Unet, self).__init__()
        self.orange1 = orange(32, 32)
        self.orange2 = orange(64, 64)
        self.orange3 = orange(128, 128)
        self.orange4 = orange(256, 256)
        self.orange5 = orange(128, 128)
        self.orange6 = orange(64, 64)
        self.orange7 = orange(32, 32)
        # self.orange1 = orange(32,32)

    def forward(self, x):
        x1_in = nn.Conv2d(1, 32, 3, padding=1)(input_image)
        x1_in = nn.BatchNorm2d(32)(x1_in)
        x = self.orange1(x1_in)
        x1_out = torch.add(x, x1_in)

        x2_in = nn.Conv2d(32, 64, 3, stride=2, padding=1)(x1_out)
        x2_in = nn.ReLU(inplace=True)(x2_in)
        x2_in = nn.BatchNorm2d(64)(x2_in)
        x = self.orange2(x2_in)
        x2_out = torch.add(x, x2_in)

        x3_in = nn.Conv2d(64, 128, 3, stride=2, padding=1)(x2_out)
        x3_in = nn.ReLU(inplace=True)(x3_in)
        x3_in = nn.BatchNorm2d(128)(x3_in)
        x = self.orange3(x3_in)
        x3_out = torch.add(x, x3_in)

        x4_in = nn.Conv2d(128, 256, 3, stride=2, padding=1)(x3_out)
        x4_in = nn.ReLU(inplace=True)(x4_in)
        x4_in = nn.BatchNorm2d(256)(x4_in)
        x = self.orange4(x4_in)
        x4_out = torch.add(x, x4_in)

        #### decoder

        # y3_in = nn.ConvTranspose2d(256, 128, 2, stride=2)(x4_out)
        # y3_in = nn.ReLU(inplace=True)(y3_in)
        # y3_in = nn.BatchNorm2d(128)(y3_in)
        # x = self.orange5(y3_in)
        # cat1 = torch.cat([x3_out, y3_in], 0)

        # y3_in = nn.Conv2d(256, 128, 3, padding=1)(cat1)
        # y3_in = nn.ReLU(inplace=True)(y3_in)
        # y3_in = nn.BatchNorm2d(128)(y3_in)
        # x = self.orange5(y3_in)
        # y3_out = torch.add(x, y3_in)

        y3_in = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(x4_out)
        y3_in = nn.ReLU(inplace=True)(y3_in)
        y3_in = nn.BatchNorm2d(128)(y3_in)
        cat1 = torch.cat([x3_out, y3_in], 0)
        y3_in = nn.Conv2d(128, 128, kernel_size=3, padding=1)(cat1)
        y3_in = nn.BatchNorm2d(128)(y3_in)
        x = self.orange5(y3_in)
        y3_out = x + y3_in
        print(y3_out.shape)

        ##########################################

        y2_in = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(y3_out)
        y2_in = nn.ReLU(inplace=True)(y2_in)
        y2_in = nn.BatchNorm2d(64)(y2_in)
        cat2 = torch.cat([x2_out, y2_in], 0)
        y2_in = nn.Conv2d(64, 64, kernel_size=3, padding=1)(cat2)
        y2_in = nn.BatchNorm2d(64)(y2_in)
        x = self.orange6(y2_in)
        y2_out = x + y2_in
        print(y2_out.shape)

        # y2_in = nn.ConvTranspose2d(128, 64, 2, stride=2)(y3_out)
        # y2_in = nn.ReLU(inplace=True)(y2_in)
        # y2_in = nn.BatchNorm2d(64)(y2_in)
        # cat2 = torch.cat(x2_out, y2_in)

        # y2_in = nn.Conv2d(128, 64, 3, padding=1)(cat2)
        # y2_in = nn.ReLU(inplace=True)(y2_in)
        # y2_in = nn.BatchNorm2d(64)(y2_in)
        # x = self.orange6(y2_in)
        # y2_out = torch.add(x, y2_in)

        ##################################################

        y1_in = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)(y2_out)
        y1_in = nn.ReLU(inplace=True)(y1_in)
        cat3 = torch.cat([x1_out, y1_in], 0)
        y1_in = nn.Conv2d(32, 32, kernel_size=3, padding=1)(cat3)
        y1_in = nn.BatchNorm2d(32)(y1_in)
        x = self.orange7(y1_in)
        y1_out = x + y1_in
        print(y1_out.shape)

        # y1_in = nn.ConvTranspose2d(64, 32, 2, stride=2)(y2_out)
        # y1_in = nn.ReLU(inplace=True)(y1_in)
        # y1_in = nn.BatchNorm2d(32)(y1_in)
        # cat3 = torch.cat(x1_out, y1_in)

        # y1_in = nn.Conv2d(64, 32, 3, padding=1)(cat3)
        # y1_in = nn.ReLU(inplace=True)(y1_in)
        # y1_in = nn.BatchNorm2d(32)(y1_in)
        # x = self.orange7(y1_in)
        # y1_out = torch.add(x, y1_in)

        ########################################################

        y_out = nn.Conv2d(32, 64, kernel_size=3, padding=1)(y1_out)
        y_out = nn.ReLU(inplace=True)(y_out)
        y_out = nn.BatchNorm2d(64)(y_out)
        print(y_out.shape)
        y_out = nn.Conv2d(64, 1, kernel_size=1)(y_out)
        print(y_out.size())

        return y_out


if __name__ == "__main__":
    model = Dense_Unet(1, 1)
    input_image = torch.rand(size=(1, 1, 32, 32))
    out = model(input_image)
    # print(out)


# class orange(nn.Module):
#     """"""

#     def __init__(self, in_ch, out_ch):
#         super(orange, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 3, padding="same")
#         self.relu = nn.ReLU(inplace=True)
#         self.BatchNorm2d = nn.BatchNorm2d(out_ch)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)

#         x = self.conv(x)
#         x = self.relu(x)

#         x = self.BatchNorm2d(x)
#         return x


# class Dense_Unet(nn.Module):
#     """"""

#     def __init__(self, in_ch, out_ch):
#         super(Dense_Unet, self).__init__()
#         self.orange1 = orange(1, 32)

#     def forward(self, x):
#         x1_in = nn.Conv2d(1, 32, 3, padding="same")(input_image)
#         x1_in = nn.BatchNorm2d(32)(x1_in)
#         x = self.orange1(x1_in)
#         x1_out = torch.add([x, x1_in])

#         x2_in = nn.Conv2D(32, 64, 2, stride=2, padding="same")(x1_out)
#         x2_in = nn.relu(inplace=True)(x2_in)
#         x2_in = nn.BatchNorm2d(64)(x2_in)
#         x = self.orange(x2_in)
#         x2_out = torch.add([x, x2_in])

#         x3_in = nn.Conv2D(64, 128, 2, stride=2, padding="same")(x2_out)
#         x3_in = nn.relu(inplace=True)(x3_in)
#         x3_in = nn.BatchNorm2d(128)(x3_in)
#         x = orange(x3_in)
#         x3_out = torch.add([x, x3_in])

#         x4_in = nn.Conv2D(128, 256, 2, stride=2, padding="same")(x3_out)
#         x4_in = nn.relu(inplace=True)(x4_in)
#         x4_in = nn.BatchNorm2d(256)(x4_in)
#         x = orange(x4_in)
#         x4_out = torch.add([x, x4_in])

#         y3_in = nn.ConvTranspose2d(256, 128, 2, stride=2)(x4_out)
#         y3_in = nn.relu(inplace=True)(y3_in)
#         y3_in = nn.BatchNorm2d(128)(y3_in)
#         cat1 = torch.cat([x3_out, y3_in])

#         y3_in = nn.Conv2D(256, 128, 3, padding="same")(cat1)
#         y3_in = nn.relu(inplace=True)(y3_in)
#         y3_in = nn.BatchNorm2d(128)(y3_in)
#         x = orange(y3_in)
#         y3_out = torch.add([x, y3_in])

#         y2_in = nn.ConvTranspose2d(128, 64, 2, stride=2)(y3_out)
#         y2_in = nn.relu(inplace=True)(y2_in)
#         y2_in = nn.BatchNorm2d(64)(y2_in)
#         cat2 = torch.cat([x2_out, y2_in])

#         y2_in = nn.Conv2D(128, 64, 3, padding="same")(cat2)
#         y2_in = nn.relu(inplace=True)(y2_in)
#         y2_in = nn.BatchNorm2d(64)(y2_in)
#         x = orange(y2_in)
#         y2_out = torch.add([x, y2_in])

#         y1_in = nn.ConvTranspose2d(64, 32, 2, stride=2)(y2_out)
#         y1_in = nn.relu(inplace=True)(y1_in)
#         y1_in = nn.BatchNorm2d(32)(y1_in)
#         cat3 = torch.cat([x1_out, y1_in])

#         y1_in = nn.Conv2D(64, 32, 3, padding="same")(cat3)
#         y1_in = nn.relu(inplace=True)(y1_in)
#         y1_in = nn.BatchNorm2d(32)(y1_in)
#         x = orange(y1_in)
#         y1_out = torch.add([x, y1_in])

#         y_out = nn.Conv2D(32, 64, 5, padding="same")(y1_out)
#         y_out = nn.BatchNorm2d(64)(y_out)
#         y_out = nn.Conv2D(64, 1)(y1_out)

#         return y_out


# if __name__ == "__main__":
#     model = Dense_Unet(1,)
#     input_image = torch.rand(size=(1, 1, 128, 128))
#     out = model(input_image)
#     print(out.shape)

# test = Dense_Unet(1, 32)


# if __name__ == "__main__":
# input_image = torch.rand(size=(8, 1, 128, 128))
# model = Dense_Unet(1, 32)
# # out = model(input_image, y1_out)
# print("")


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
