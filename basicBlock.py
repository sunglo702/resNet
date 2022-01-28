import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable

def con3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

def con1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

# define basicBlock
class basicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(basicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError ('BasicBlock only supports groups=1 and base_width=64')# exception value error
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")# exception not implemented error

        # define the layers about basicBlock
        self.conv1 = con3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        #
        self.conv2 = con3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.dowansample = downsaple
        self.stride = stride

    # connect forward layers which is had defined
    def forward(self, x):
        identity = x # resiual block need to save origin input

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # to ensure origin input and convoluted out layers dimension are the same
        if self.dowansample is not None:
            identity = self.dowansample(x)
        out += identity
        out = self.relu(out)

        return out

