import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
import torchvision.models as models

class SSD(nn.Module):
    """
    Args:
    is_train(bool): True if train. False if test
    base_network(?): base network for SSD (VGG in the paper) 
    extra_layers(?): extra layers to perform detect from multi-scale feature maps 
    objectives: 고민
    """
    def __init__(self, is_train, base_network, objectives=create_output(), num_classes=201, image_size=300):
        super(SSD, self).__init__()
        self.is_train = is_train
        self.base_network = nn.ModuleList(list(base_network.features))
        self.num_classes = num_classes
        self.image_size = image_size
        
        self.L2Normalization = L2Normalization(512)
        

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(1024,256,kernel_size=(1,1),stride=1),
                                nn.Conv2d(256,512, kernel_size=3, stride=2, padding=1)))

        self.layers.append( nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1),
                               nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)))
        self.layers.append( nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1),
                               nn.Conv2d(128, 256, kernel_size=3, stride=1)))
        self.layers.append( nn.Sequential(nn.Conv2d(256,128, kernel_size=1,stride=1),
                               nn.Conv2d(128,256, kernel_size=3, stride=1)))

        self.locations, self.confidences = objectives
        
    def forward(self, x):
        """
        Args:
            x(torch.Tensor): input image, shape = (batch_size, 3, 300, 300)
            
        Returns:
            train phase:
            test phase:
        """
        feature_map = []
        loc_output = []
        conf_output =[]
        for i in range(23):
            x = self.base_network[i](x) # shape = (batch_size, 512, 38, 38)
        
        # 38 * 38 feature map -> L2 Norm
        y = self.L2Normalization(x) 
        feature_map.append(y)
        
        for i in range(23, 35):
            x = self.base_network[i](x) # shape = (batch_size, 1024, 19, 19)
        feature_map.append(x)
        # extra 4 heads
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        for x, l, c in zip(feature_map, self.locations, self.confidences):
            loc_output.append(l(x))
            conf_output.append(c(x))
            
        return loc_output, conf_output
        # output = confidence, locations 

        # return output
    # model loading

    # multibox

    # add extra

    # build ssd?

# L2 norm Implementation




class L2Normalization(nn.Module):
    """
    L2 normalization for conv4_3(VGG 16)
    args:
        nc: number of channels
    """
    def __init__(self, nc,scale=20):
        super(L2Normalization, self).__init__()
        self.parameter = nn.Parameter(torch.Tensor(1, nc,1, 1)) # [batch_size, channel, w, h] = [1, channel, 1, 1]
        nn.init.constant_(self.parameter, scale)
        
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True) # L2 Norm of feature map x (Channel wise?)
        x = x.div(norm)
        x = self.parameter.expand_as(x) * x
        return x

# confidence, location
def create_output(channels=[512, 1024,512,256,256,256], num_boxes=[4, 6, 6, 6, 4, 4],num_classes=201):
    locations = []
    confidences = []
    for channel, num_box in zip(channels, num_boxes):
        confidences += [nn.Conv2d(channel, num_box* num_classes, kernel_size=3,padding=1)]
        locations += [nn.Conv2d(channel, num_box * 4, kernel_size=3, padding=1)]
    return locations, confidences

