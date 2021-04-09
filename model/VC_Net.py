from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision import models
from model.res2net.model.res2net import res2net50

nn.AdaptiveAvgPool2d(1)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = F.interpolate(x, size=None,scale_factor=2, align_corners=True)
        x = F.interpolate(x, size=None,scale_factor=2)
        x = self.up(x)
        return x

class VC_Net(nn.Module):
    def __init__(self,  in_ch=3, out_ch=3, is_pretrained=True):
        super(VC_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2 * 2, n1 * 4 * 2, n1 * 8 * 2, n1 * 16 * 2]

        res2net = res2net50(pretrained=is_pretrained)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = res2net.layer1#torch.Size([2, 256, 256, 256])
        self.encoder2 = res2net.layer2#torch.Size([2, 512, 128, 128])
        self.encoder3 = res2net.layer3#torch.Size([2, 1024, 64, 64])
        self.encoder4 = res2net.layer4#torch.Size([2, 2048, 32, 32])

        self.Conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[3] + filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[2] + filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[1] + filters[1], filters[1])

        self.Up2 = up_conv(filters[1], int(filters[1] / 2))
        self.Up_conv2 = conv_block(int(filters[1] / 2) + filters[0], filters[0])
        self.Up_conv21 = conv_block(filters[0], int(filters[0] / 2))#32

        self.conv_v = conv_block(int(filters[0] / 2), int(filters[0] / 2))
        self.conv_v_1 = nn.Conv2d(int(filters[0] / 2), 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.conv_av = conv_block(int(filters[0] / 2), int(filters[0] / 2))


        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        e0 = self.Conv1(x)
        xm = self.Maxpool1(e0)#64-256

        e1 = self.encoder1(xm)#256-256
        e2 = self.encoder2(e1)#512-128
        e3 = self.encoder3(e2)#1024-64

        e4 = self.encoder4(e3)#2048-32

        #centeer
        e5 = self.Conv5(e4)

        # Decoder

        d4 = self.Up5(e5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv5(d4)

        d3 = self.Up4(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv4(d3)

        d2 = self.Up3(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv3(d2)

        d1 = self.Up2(d2)
        d1 = torch.cat((e0, d1), dim=1)
        d1 = self.Up_conv2(d1)
        d1 = self.Up_conv21(d1)

        d0_v = self.conv_v(d1)
        d0_v1 = self.conv_v_1(d0_v)
        d0_v1_bn = self.bn(d0_v1)
        d0_v1_bn_s = F.sigmoid(d0_v1_bn)

        # d0_v1_bn_s_5_e = (torch.exp(-torch.pow(d0_v1_bn_s - 0.5, 1)) - torch.exp(-1/2))+ 1
        d0_v1_bn_s_5_e = (torch.exp(-torch.abs(d0_v1_bn_s - 0.5)) - torch.exp(torch.tensor(-1/2)))+ 1


        d0_av = self.conv_av(d1)


        d0_cat = torch.cat((d0_av, d0_v), dim=1)
        d0_ = d0_cat * d0_v1_bn_s_5_e * d0_v1_bn_s
        d0 = self.Conv(d0_)

        return d0, d0_v1_bn_s * d0_v1_bn_s_5_e
