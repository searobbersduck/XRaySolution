import os
import sys
sys.path.append('../')
sys.path.append('../train/')
from resnet import *
import torch
from dr_model import DRModel


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='convert pytorch model to onnx')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--outname', default='dr.onnx')
    return parser.parse_args()

opt = parse_args()
print('\n====>opt:\t')
print(opt)
print('\n')
# model = resnet34(num_classes=6, shortcut_type=True, sample_size=128, sample_duration=128)
model = DRModel('rsn34', 1024, 2)

# weights = '../train/model/dr_cls_yyy/ct_pos_recognition_0020_best.pth'
weights = opt.weights
model.load_state_dict(torch.load(weights))

dummy_input = torch.randn(1,1,1024,1024)

outname = opt.outname
torch.onnx.export(model, dummy_input, outname, verbose=True, input_names=['input'], output_names=['output'])

print('====> export to onnx model!')

