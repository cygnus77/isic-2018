#pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    num_classes = 1
    
    def __init__(self):
        super(Net, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = vgg16.features
        for i,param in enumerate(self.encoder.parameters()):
            param.requires_grad = i >= 16
        #self.decoder1x1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.a_convT2d = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.a_relu1 = nn.ReLU(inplace=True)
        self.a_conv2d1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.a_relu2 = nn.ReLU(inplace=True)
        self.a_conv2d2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.a_relu3 = nn.ReLU(inplace=True)
        
        
        self.b_convT2d = nn.ConvTranspose2d(in_channels=768, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.b_relu1 = nn.ReLU(inplace=True)
        self.b_conv2d1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b_relu2 = nn.ReLU(inplace=True)
        self.b_conv2d2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b_relu3 = nn.ReLU(inplace=True)

        self.convT2d3 = nn.ConvTranspose2d(in_channels=384, out_channels=1, kernel_size=4, stride=4, padding=0)
    
    def forward(self, x):

        skipConnections = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [30, 23, 15]:
                skipConnections[i] = x
                
        #x = self.decoder1x1(x)
        # x = torch.cat((x,skipConnections[30]), 1)
        x = self.a_convT2d(x)
        x = self.a_relu1(x)
        x = self.a_conv2d1(x)
        x = self.a_relu2(x)
        x = self.a_conv2d2(x)
        x = self.a_relu3(x)
        
        x = torch.cat((x,skipConnections[23]), 1)
        
        x = self.b_convT2d(x)
        x = self.b_relu1(x)
        x = self.b_conv2d1(x)
        x = self.b_relu2(x)
        x = self.b_conv2d2(x)
        x = self.b_relu3(x)

        x = torch.cat((x, skipConnections[15]), 1)

        x = self.convT2d3(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size()[0], -1, Net.num_classes)
        
        return x

