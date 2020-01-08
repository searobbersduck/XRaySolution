import os
import numpy as np
import pickle
from tensorrtserver.api import *
import cv2
import time

protocol = ProtocolType.from_str('http')
# ctx = InferContext('10.100.37.20:19992', protocol, 'dr_cls', -1, True)
ctx = InferContext('10.100.37.20:19992', protocol, 'feibuzhang_cls', -1, True)

def get_img(infile):
    img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32)
    img = img/255
    img = np.expand_dims(img, 0)
    return img


test_file = '/data/zhangwd/data/examples/dr_deformable/test_label.txt'

test_files = []
test_flags = []
with open(test_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line is None or len(line) == 0:
            continue
        ss = line.split('\t')
        if len(ss) != 2:
            continue
        test_files.append(ss[0])
        test_flags.append(int(ss[1]))

print(test_files)
pred_flags = []
for infile in test_files:
    img = get_img(infile)
    result = ctx.run({'images': (img,)}, {'result': (InferContext.ResultFormat.CLASS, 2)}, 1)
    print(result['result'])
    print(int(result['result'][0][0][0]))
    pred_flags.append(int(result['result'][0][0][0]))

print('====> pred:\t')
print(pred_flags)
print('====> gt:\t')
print(test_flags)

np_test_flags = np.array(test_flags, dtype=np.int8)
np_pred_flags = np.array(pred_flags, dtype=np.int8)

print('====> accuracy:\t{:.3f}'.format(np.sum(np_test_flags==np_pred_flags)/len(test_flags)))


def infer_with_trt(test_file, service_ip, service_port, service_name, out_dir):
    protocol = ProtocolType.from_str('http')
    # ctx = InferContext('10.100.37.20:19992', protocol, 'dr_cls', -1, True)
    # ctx = InferContext('10.100.37.20:19992', protocol, 'feibuzhang_cls', -1, True)
    ctx = InferContext('{}:{}'.format(service_ip, service_port), protocol, service_name, -1, True)

    # test_file = '/data/zhangwd/data/examples/dr_deformable/test_label.txt'

    test_files = []
    test_flags = []
    with open(test_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if len(ss) != 2:
                continue
            test_files.append(ss[0])
            test_flags.append(int(ss[1]))

    print(test_files)
    pred_flags = []
    time_data = 0
    time_model = 0
    time_total = 0
    time_beg = time.time()
    for infile in test_files:
        time1 = time.time()
        img = get_img(infile)
        time2 = time.time()
        time_data += (time2-time1)
        result = ctx.run({'images': (img,)}, {'result': (InferContext.ResultFormat.CLASS, 2)}, 1)
        time3 = time.time()
        time_model += (time3-time2)
        print(result['result'])
        print(int(result['result'][0][0][0]))
        pred_flags.append(int(result['result'][0][0][0]))
    time_end = time.time()
    time_total = time_end - time_beg

    print('====> pred:\t')
    print(pred_flags)
    print('====> gt:\t')
    print(test_flags)

    np_test_flags = np.array(test_flags, dtype=np.int8)
    np_pred_flags = np.array(pred_flags, dtype=np.int8)

    print('====> accuracy:\t{:.3f}'.format(np.sum(np_test_flags==np_pred_flags)/len(test_flags)))