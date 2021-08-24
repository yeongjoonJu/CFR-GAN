import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable

class AFD(nn.Module):
    """
    Adaptive Feature Difference Module
    """
    def __init__(self, in_dim, out_dim):
        super(AFD, self).__init__()

        self.embedding_r = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.embedding_g = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.conv_IN = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dim)
        )
    
    def forward(self, r, g):

        diff = torch.square(r-g)
        return self.conv_IN(diff)


class AFD_prev(nn.Module):
    """
    Adaptive Feature Difference Module
    """
    def __init__(self, in_dim, out_dim):
        super(AFD_prev, self).__init__()

        self.embedding_r = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.embedding_g = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.conv_IN = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dim)
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, r, g):
        diff1 = torch.square(r-g)
        r = self.embedding_r(r)
        g = self.embedding_g(g)
        diff2 = torch.square(r-g)

        return self.conv_IN(diff1 + self.upsample(diff2))


class MixingLayer(nn.Module):
    def __init__(self, dims=[64, 128, 256]):
        super(MixingLayer, self).__init__()

        self.upsample_64_to_128 = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dims[1]), nn.LeakyReLU(0.2, True)
        )
        self.upsample_128_to_256 = nn.Sequential(
            nn.Conv2d(dims[1]*2, dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[1]), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(dims[1], dims[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dims[0]), nn.LeakyReLU(0.2, True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=3, padding=1), nn.Sigmoid()
        )
    def forward(self, x1, x2, x3):
        x3 = self.upsample_64_to_128(x3)
        x2 = self.upsample_128_to_256(torch.cat([x2,x3], dim=1))
        return self.conv(x1 + x2)


class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class SNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SNConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        return self.conv2d(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            SNConv(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim), nn.LeakyReLU(0.2,True),
            nn.ReflectionPad2d(1),
            SNConv(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )
        
    def forward(self, x):
        return x + self.conv_block(x)


class ResnetBlock_prev(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock_prev, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            SNConv(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim), nn.LeakyReLU(0.2,True),
            nn.ReflectionPad2d(1),
            SNConv(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )
        self.gating = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim), nn.Sigmoid()
        )
        
    def forward(self, x):
        mask = self.gating(x)
        return x*mask + self.conv_block(x)