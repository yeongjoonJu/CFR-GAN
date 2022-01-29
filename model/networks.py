import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16, vgg19, resnet18
from model.layers import *


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # if not requires_grad:
        #     for param in self.parameters():
        #         param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
        
class ExpressionClassifier(nn.Module):
    def __init__(self, backbone_name='vgg16'):
        super(ExpressionClassifier, self).__init__()
        if backbone_name == 'vgg16':
            self.backbone = vgg16(pretrained=True)
        elif backbone_name == 'res18':
            self.backbone = resnet18(pretrained=True)
        else:
            raise NotImplementedError
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)          
        )
    
    def forward(self, img):
        features = self.backbone(img)
        features = features.view(img.shape[0], -1)

        return self.fc(features)


class CFRNet(nn.Module):
    """
    Complete Face Recovery Model
    """
    def __init__(self):
        super(CFRNet, self).__init__()
        self.conv_r = nn.Sequential(
            nn.ReflectionPad2d(3),
            # nn.Conv2d(3, 64, kernel_size=7, padding=0),
            SNConv(3, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
        )
        self.conv_g = nn.Sequential(
            nn.ReflectionPad2d(3),
            # nn.Conv2d(3, 64, kernel_size=7, padding=0),
            SNConv(3,64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
        )

        self.afd = AFD(64, 64)
        
        self.downsample = nn.ModuleList([
            StridedGatedConv(64, 128, kernel_size=3), # 128
            StridedGatedConv(128, 256, kernel_size=3), # 64
        ])


        blocks = []
        for _ in range(9):
            blocks.append(ResnetBlock(256, gate=True))
        self.resblocks = nn.Sequential(*blocks)

        self.upsample = nn.Sequential(
            ConvTransINAct(256, 128, ks=4, stride=2, padding=1),
            ConvTransINAct(128, 64, ks=4, stride=2, padding=1),
        )

        self.last_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            # nn.Conv2d(64, 3, kernel_size=7, padding=0),
            SNConv(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.mixing = MixingLayer(dims=[64, 128, 256])

        self.sigmoid = nn.Sigmoid()

        self.init_weight()
    
    def forward(self, rotated, guidance, wo_mask=False):
        # Normalize
        rotated = self.conv_r(rotated)
        guidance = self.conv_g(guidance)

        diff = self.afd(rotated, guidance)
        attn = self.sigmoid(diff)

        out, gate1 = self.downsample[0](rotated*(1.-attn)) # 128
        out, gate2 = self.downsample[1](out) # 64

        out = self.resblocks(out)
        out = self.upsample(out)
        out = self.last_conv(out)
        
        if wo_mask:
            return out

        mask = self.mixing(diff, gate1, gate2)
        
        return out, mask
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SNConvWithActivation(3, 64, 4, 2, padding=2), #
            SNConvWithActivation(64, 128, 4, 2, padding=2), #  / 56
            SNConvWithActivation(128, 256, 4, 2, padding=2), # 32 / 28
            SNConvWithActivation(256, 512, 4, 2, padding=2),
            SNConv(512, 1, kernel_size=1, stride=1, padding=0)

        )
        self.init_weight()

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return out
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class TwoScaleDiscriminator(nn.Module):
    def __init__(self):
        super(TwoScaleDiscriminator, self).__init__()

        self.D1 = Discriminator()
        self.D2 = Discriminator()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):        
        out1 = self.D1(x)
        out2 = self.D2(self.downsample(x))

        return [out1, out2]