#pylint: disable=E1101
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    num_classes = 1
    
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        for i,param in enumerate(self.resnet.parameters()):
            print(param)
            param.requires_grad = False

        self.a_convT2d = nn.ConvTranspose2d(in_channels=2048, out_channels=256, kernel_size=4, stride=2, padding=1)              
        self.b_convT2d = nn.ConvTranspose2d(in_channels=1280, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.convT2d3 = nn.ConvTranspose2d(in_channels=384, out_channels=1, kernel_size=4, stride=4, padding=0)
    
    def freeze(self, n):
        for i,param in enumerate(self.resnet.parameters()):
                param.requires_grad = i >= n
    
    def forward(self, x):

        skipConnections = {}
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        skipConnections[1] = x = self.resnet.layer1(x)   # [10, 256, 56, 56]
        
        skipConnections[2] = x = self.resnet.layer2(x) # [10, 512, 28, 28]
        
        skipConnections[3] = x = self.resnet.layer3(x) # [10, 1024, 14, 14]
        
        skipConnections[4] = x = self.resnet.layer4(x) # [10, 2048, 7, 7]

        x = self.a_convT2d(x)  # [10, 256, 14, 14]

        x = torch.cat((x,skipConnections[3]), 1)
        
        x = self.b_convT2d(x) # [10, 128, 56, 56]

        x = torch.cat((x, skipConnections[1]), 1)

        x = self.convT2d3(x) # [10, 1, 224, 224]

        x = nn.Sigmoid()(x)
        x = x.view(x.size()[0], -1, Net.num_classes)
        
        return x

