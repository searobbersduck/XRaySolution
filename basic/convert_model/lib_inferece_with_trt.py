import os
import numpy as np
import pickle
from tensorrtserver.api import *
import cv2
import time
import json

import xlrd
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x

def plot_roc(y_true, y_pred):
    log = []
    from sklearn import metrics
    def calc_metrics_table(y_true, y_pred, thresholds):
        metrics_list = list()
        for threshold in thresholds:
            y_pred_binary = np.zeros(y_pred.shape, dtype=np.uint8)
            y_pred_binary[y_pred>threshold] = 1
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
            print('tn:{:.3f}\tfp:{:.3f}\tfn:{:.3f}\ttp:{:.3f}\t'.format(tn, fp, fn, tp))
            accuracy = (tp+tn)/(tn+fp+fn+tp)
            sensitivity = tp/(tp+fn)
            specificity = tn/(fp+tn)
            ppv = tp/(tp+fp)
            npv = tn/(tn+fn)
            metrics_list.append([threshold, accuracy, sensitivity, specificity, ppv, npv])
        metrics_table = pd.DataFrame(np.array(metrics_list), columns=['threshold','accuracy','sensitivity','specificity','ppv','npv'])
        return metrics_table


    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred)

    auc = metrics.auc(fpr, tpr)

    thresholds = np.arange(0.05, 1., 0.05)
    metrics_table = calc_metrics_table(y_true, y_pred, thresholds)

    print('AUC:%.4f'% auc)
    log.append('AUC:%.4f'% auc)

    plt.title('roc curve')
    plt.plot(fpr, tpr, 'r')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(ls='--')
    # plt.show()

    print(metrics_table)
    log.append(metrics_table)
    return log

def get_img(infile):
    img = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32)
    img = img/255
    img = np.expand_dims(img, 0)
    return img

def infer_with_trt(test_file, service_ip, service_port, service_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
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
        probs = [0,0]
        probs[int(result['result'][0][0][0])] = result['result'][0][0][1]
        probs[int(result['result'][0][1][0])] = result['result'][0][1][1]
        probs_s = softmax(np.array(probs))
        pred_flags.append(probs_s[1])
        # pred_flags.append(int(result['result'][0][0][0]))
    time_end = time.time()
    time_total = time_end - time_beg

    print('====> pred:\t')
    print(pred_flags)
    print('====> gt:\t')
    print(test_flags)

    np_test_flags = np.array(test_flags, dtype=np.int8)
    np_pred_flags = np.array(pred_flags, dtype=np.int8)

    print('====> accuracy:\t{:.3f}'.format(np.sum(np_test_flags==np_pred_flags)/len(test_flags)))

    print('====> data time:\t{:.3f}'.format(time_data))
    print('====> model time:\t{:.3f}'.format(time_model))
    print('====> total time:\t{:.3f}'.format(time_total))

    out_result_file = os.path.join(out_dir, '{}_result.txt'.format(service_name))
    out_log_file = os.path.join(out_dir, '{}_log.txt'.format(service_name))

    log = plot_roc(np.array(test_flags, dtype=np.float32), np.array(pred_flags))

    with open(out_log_file, 'w') as f:
        # f.write('\n'.join(log))
        f.write('{}'.format(log[0]))

    with open(out_result_file, 'w') as f:
        for i in range(len(pred_flags)):
            f.write('{}\t{}\n'.format(test_flags[i], pred_flags[i]))



# infer_with_trt('/data/zhangwd/data/examples/dr_deformable/test_label.txt', '10.100.37.20', '19992', 'feidapao_cls', 'inference_result')

def infer_task(task_name):
    config_file = os.path.join('../train/config/config_dr_{}.json'.format(task_name))
    service_name = '{}_cls'.format(task_name)
    with open(config_file, encoding='gb2312') as f:
        config = json.load(f)
    infer_with_trt(config['test_list_file'], '10.100.37.20', '19993', service_name, 'inference_result')

# infer_task('dr')
# infer_task('feidapao')
infer_task('qixiong')
# infer_task('shuhou')
# infer_task('xianwei')
# infer_task('xiongqiangjiye')
# infer_task('zhanwei')
# infer_task('feibuzhang')
# infer_task('feiqizhong')
# infer_task('shibian')
# infer_task('wenli')
# # infer_task('xinying')
# infer_task('yanxing')
# infer_task('zhudongmai')
