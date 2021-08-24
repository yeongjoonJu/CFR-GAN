import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16, vgg19, resnet50, resnet18
from layers import *

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
