import os
import numpy as np
import sys
# import scipy.ndimage as nd
import json
# import pickle
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from resnet import *
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import math
from utils import AverageMeter
# from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()


class DRModel(nn.Module):
    def __init__(self, name, inmap, multi_classes, weights=None, scratch=False):
        super(DRModel, self).__init__()
        self.name = name
        self.weights = weights
        self.inmap = inmap
        self.multi_classes = multi_classes
        self.featmap = inmap // 32
        self.planes = 2048
        base_model = None
        if name == 'rsn18':
            base_model = resnet18()
            self.planes = 512
        elif name == 'rsn34':
            base_model = resnet34()
            self.planes = 512
        elif name == 'rsn50':
            base_model = resnet50()
            self.planes = 2048
        elif name == 'rsn101':
            base_model = resnet101()
            self.planes = 2048
        elif name == 'rsn152':
            base_model = resnet152()
            self.planes = 2048

        #         if not scratch:
        #             base_model.load_state_dict(torch.load('../pretrained/'+name+'.pth'))

        self.base = nn.Sequential(*list(base_model.children())[:-2])
        if name == 'rsn18' or name == 'rsn34' or name == 'rsn50' or name == 'rsn101' or name == 'rsn152':
            self.base = nn.Sequential(*list(base_model.children())[:-2])
        elif name == 'dsn121' or name == 'dsn161' or name == 'dsn169' or name == 'dsn201':
            self.base = list(base_model.children())[0]

        self.cls = nn.Linear(self.planes, multi_classes)

        initial_cls_weights(self.cls)

        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        feature = self.base(x)
        # when 'inplace=True', some errors occur!!!!!!!!!!!!!!!!!!!!!!
        out = F.relu(feature, inplace=False)
        out = F.avg_pool2d(out, kernel_size=self.featmap).view(feature.size(0), -1)
        out = self.cls(out)
        return out