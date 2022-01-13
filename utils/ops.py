import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
import math

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def sobel_xy(image, norm='L1'):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)

    kernel_x = torch.FloatTensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]])
    #kernel_x = torch.FloatTensor([[[[3,0,-3],[10,0,-10],[3,0,-3]]]])
    kernel_y = torch.FloatTensor([[[[1,2,1],[0,0,0],[-1,-2,-1]]]])
    #kernel_y = torch.FloatTensor([[[[3,10,3],[0,0,0],[-3,-10,-3]]]])

    gradient_x = F.conv2d(image, kernel_x.cuda(), padding=1)
    gradient_y = F.conv2d(image, kernel_y.cuda(), padding=1)

    if norm=='L1':
        gradient = torch.abs(gradient_x) + torch.abs(gradient_y)
    elif norm=='L2':
        gradient = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))

    return gradient


def blur(image, filter_size=5):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)

    channels = image.shape[1]
    kernel = torch.ones(1, 1, filter_size, filter_size) / (filter_size*filter_size)

    out = None
    padding = (filter_size-1)//2
    for channel in range(channels):
        _out = F.conv2d(image[:,channel,...].unsqueeze(1), kernel.cuda(), padding=padding)
        if out is None:
            out = _out
        else:
            out = torch.cat([out, _out], dim=1)
    
    return out


def erosion(image, filter_size=5):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)
        
    pad_total = filter_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    image = F.pad(image, (pad_beg, pad_end, pad_beg, pad_end))
    kernel = torch.zeros(1, 1, filter_size, filter_size).to(image.device)
    image = F.unfold(image, filter_size, dilation=1, padding=0, stride=1)
    image = image.unsqueeze(1)
    L = image.size(-1)
    L_sqrt = int(math.sqrt(L))

    kernel = kernel.view(1, -1)
    kernel = kernel.unsqueeze(0).unsqueeze(-1)
    image = kernel - image
    image, _ = torch.max(image, dim=2, keepdim=False)
    image = -1 * image
    image = image.view(-1, 1, L_sqrt, L_sqrt)

    return image


def dilation(image, filter_size=7):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)
        
    pad_total = filter_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    image = F.pad(image, (pad_beg, pad_end, pad_beg, pad_end))
    kernel = torch.zeros(1, 1, filter_size, filter_size).to(image.device)
    image = F.unfold(image, filter_size, dilation=1, padding=0, stride=1)
    image = image.unsqueeze(1)
    L = image.size(-1)
    L_sqrt = int(math.sqrt(L))

    kernel = kernel.view(1, -1)
    kernel = kernel.unsqueeze(0).unsqueeze(-1)
    image = kernel + image
    image, _ = torch.max(image, dim=2, keepdim=False)
    image = image.view(-1, 1, L_sqrt, L_sqrt)

    return image


def rgb_to_lab(srgb):
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ]).type(torch.FloatTensor).cuda()

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)


    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).cuda())

    epsilon = 6.0/29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).cuda()

    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).cuda()

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ]).type(torch.FloatTensor).cuda()
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()

    return torch.reshape(lab_pixels, srgb.shape)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # luminance = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    # contrast = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # structure = (2*sigma12 + C2)/(2*sigma1_2 + C2)
    con_str = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)

    return con_str


class SCDiffer(nn.Module):
    """
    Structure and Color Difference
    """
    def __init__(self, filter_size=5, size_average = True):
        super(SCDiffer, self).__init__()
        self.filter_size = filter_size
        self.size_average = size_average
        self.channel = 3
        self.medianblur = MedianPool2d(kernel_size=5, padding=2)
        self.window = create_window(filter_size, self.channel).cuda()
    
    def forward(self, img1, img2, alpha=0.33):
        # self.window = window.cuda(img1.get_device()).type_as(img1)
        self.window = self.window.type_as(img1)

        con_str = _ssim(img1, img2, self.window, self.filter_size, self.channel)
        con_str = torch.mean(1.-con_str, dim=1, keepdim=True)

        img1_labs = blur(rgb_to_lab(img1), filter_size=3)
        img2_labs = blur(rgb_to_lab(img2), filter_size=3)

        lab_diff = torch.square(img1_labs - img2_labs)
        lab_diff = torch.pow(torch.sum(lab_diff, dim=1, keepdim=True), 0.5)

        con_str = (con_str-torch.mean(con_str, dim=[1,2,3], keepdim=True)) / torch.std(con_str, dim=[1,2,3], keepdim=True)
        lab_diff = (lab_diff-torch.mean(lab_diff, dim=[1,2,3], keepdim=True)) / torch.std(lab_diff, dim=[1,2,3], keepdim=True)

        # occ_map = (con_str + 1.0) * lab_diff
        occ_map = lab_diff*con_str + lab_diff + con_str*alpha
        occ_map = (occ_map - torch.mean(occ_map, dim=[1,2,3], keepdim=True)) / torch.std(occ_map, dim=[1,2,3], keepdim=True)
        occ_map = torch.clamp(occ_map, 0.0, 1.0)
        occ_map = self.medianblur(occ_map)
        # occ_map = erosion(occ_map, filter_size=2)
        
        return occ_map
