import os
import numpy as np
import sys
import scipy.ndimage as nd
import json
import pickle
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
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from dr_model import DRModel

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='dr classification')
    parser.add_argument('--config_file', required=True)
    return parser.parse_args()

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

class BinClsDataSet(torch.utils.data.Dataset):
    def __init__(self, config_file):
        self.images_list = []
        self.labels_list = []
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) != 2:
                    continue
                if not os.path.isfile(ss[0]):
                    continue
                self.images_list.append(ss[0])
                self.labels_list.append(ss[1])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#             transforms.Normalize(mean=[0], std=[255])
        ])
    def __getitem__(self, item):
        image = self.transform(Image.open(self.images_list[item]))
        return image, int(self.labels_list[item]), self.images_list[item]
    def __len__(self):
        return len(self.images_list)

def train(train_dataloader, model, criterion, optimizer, epoch, display):
    model.train()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels, _) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        _, pred = torch.max(output, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        losses.update(loss.data.cpu().numpy(), len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        if (num_iter+1) % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_dataloader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    return accuracy.avg, logger

def val(train_dataloader, model, criterion, optimizer, epoch, display):
    model.eval()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    tot_prob = np.array([], dtype=np.float32)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels,_) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        output = model(Variable(images.cuda()))
        loss = criterion(output, Variable(labels.cuda()))
        _, pred = torch.max(output, 1)
        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        tot_prob = np.append(tot_prob, F.softmax(output).cpu().detach().numpy()[:,1])
        losses.update(loss.data.cpu().numpy(), len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        if (num_iter+1) % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_dataloader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    return accuracy.avg, logger, tot_prob, tot_label

def test(train_dataloader, model, criterion, optimizer, epoch, display):
    return val(train_dataloader, model, criterion, optimizer, epoch, display)


def main():
    print('===> Parsing options')
    opt = parse_args()
    print(opt)

    config_file = opt.config_file

    config = None
    with open(config_file,encoding='gb2312') as f:
        config = json.load(f)
    print('\n')
    print('====> parse options:')
    print(config)
    print('\n')

    train_file = os.path.join(os.path.join(config["out_data_path"], 'train', 'flags.txt')) if config[
                                                                                                  "train_list_file"] == "" else \
    config["train_list_file"]
    val_file = os.path.join(os.path.join(config["out_data_path"], 'val', 'flags.txt')) if config[
                                                                                              "val_list_file"] == "" else \
    config["val_list_file"]
    print('training flags file path:\t{}'.format(train_file))
    print('validation flags file path:\t{}'.format(val_file))

    print('====> create output model path:\t')
    os.makedirs(config["model_dir"], exist_ok=True)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    time_stamp = config["export_name"]
    model_dir = os.path.join(config["model_dir"], 'dr_cls_{}'.format(time_stamp))
    os.makedirs(model_dir, exist_ok=True)

    print('====> building model:\t')

    model = DRModel('rsn34', config['scale'], config["num_classes"])
    initial_cls_weights(model)
    pretrained_weights = config['weight']
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    criterion = nn.CrossEntropyLoss().cuda()

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    if config['phase'] == 'train':
        train_ds = BinClsDataSet(train_file)
        val_ds = BinClsDataSet(val_file)
        train_dataloader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=False)
        best_acc = 0.6
        epochs = config['epoch']
        display = config['display']
        for epoch in range(epochs):
            if epoch < config['fix']:
                lr = config['lr']
            else:
                lr = config['lr'] * (0.1 ** (epoch // config['step']))
            mom = config['mom']
            wd = config['wd']
            optimizer = None
            if config['optimizer'] == 'sgd':
                optimizer = optim.SGD([{'params': model.parameters()}],
                                      lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
            print('====> train:\t')
            _, _ = train(train_dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, display)
            print('====> validate:\t')
            acc, logger, _,_ = val(val_dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, display)
            print('val acc:\t{:.3f}'.format(acc))
            if acc > best_acc:
                print('\ncurrent best accuracy is: {}\n'.format(acc))
                best_acc = acc
                saved_model_name = os.path.join(model_dir, 'ct_pos_recognition_{:04d}_best.pth'.format(epoch))
                torch.save(model.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))
    elif config['phase'] == 'test':
        print('====> begin to test:')
        test_file = os.path.join(os.path.join(config["out_data_path"], 'test', 'flags.txt')) if config[
                                                                                                    "test_list_file"] == "" else \
        config["test_list_file"]
        print('test flags file path:\t{}'.format(test_file))
        test_ds = BinClsDataSet(test_file)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=False)
        acc, logger = val(test_dataloader, nn.DataParallel(model).cuda(), criterion, 0, 10)
        print('\t====> test accuracy is {:.3f}'.format(acc))
        print('====> end to test!')

if __name__ == '__main__':
    main()