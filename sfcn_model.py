import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import module
from config import Config

epsilon = 1e-7

class identity_block(nn.Module):
    '''(Conv=>BN=>ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(identity_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y

class strided_block(nn.Module):
    '''downsample featuremap between modules'''

    def __init__(self, in_ch, out_ch):
        super(strided_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3 ,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y

class conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Res_Module, self).__init__()
        if downsample:
            self.conv1 = strided_block(in_ch, out_ch)
        else:
            self.conv1 = identity_block(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = conv_1x1(64, 2)
        self.conv3 = conv_1x1(128, 2)
        self.conv4 = conv_1x1(128, 5)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(5, 5, 2, stride=2),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(5, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 128, downsample=False)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, *input):
        x = input[0]
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        d2 = self.deconv2(s1)
        det = self.out(d2)

        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        d4 = self.deconv4(d3)
        cls = self.out(d4)
        return det, cls

class Attention_Net(nn.Module):
    def __init__(self):
        super(Attention_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = conv_1x1(64, 32)
        self.conv3 = conv_1x1(128, 32)
        self.conv4 = conv_1x1(128, 32)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 128, downsample=False)
        self.att_conv = nn.Conv2d(32, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        att1 = self.att_conv(s1)
        att1 = att1.view(att1.size(0), -1)
        att1 = self.softmax(att1)
        att1 = att1.view(att1.size(0), 1 , 128, 128)
        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        att2 = self.att_conv(d3)
        att2 = att2.view(att2.size(0), -1)
        att2 = self.softmax(att2)
        att2 = att1.view(att2.size(0), 1, 128, 128)
        d3_att = d3 * att1

        s1_att = s1 * att2
        d2 = self.deconv2(s1_att)
        det = self.out(d2)
        d4 = self.deconv4(d3_att)
        cls = self.out(d4)
        return det, cls

class Attention_Net_Global(nn.Module):
    def __init__(self):
        super(Attention_Net_Global, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = conv_1x1(64, 32)
        self.conv3 = conv_1x1(128, 32)
        self.conv4 = conv_1x1(128, 32)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 128, downsample=False)
        self.att_conv = nn.Conv2d(32, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.LogSoftmax(dim=1)
        self.att_global = module.AttentionGlobal(patch_size=16,inplane=32,outplane=1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        att1 = self.att_global(s1)
        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        att2 = self.att_global(d3)
        d3_att = d3 * att1

        s1_att = s1 * att2
        d2 = self.deconv2(s1_att)
        det = self.out(d2)
        d4 = self.deconv4(d3_att)
        cls = self.out(d4)
        return det, cls
