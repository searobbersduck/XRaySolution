import pydicom
import SimpleITK as sitk
import os
from scipy import misc
import glob
from shutil import copyfile
import numpy as np

import multiprocessing
from multiprocessing import Process
multiprocessing.freeze_support()

def resize_dr_dicom_to_size(infile, scale_size, out_dir):
    ds = pydicom.read_file(infile)
    photo_inter = ds.PhotometricInterpretation

    wc = ds.WindowCenter
    ww = ds.WindowWidth

    sitk_image = sitk.ReadImage(infile)
    sitk_image = sitk.GetArrayFromImage(sitk_image)
    image_single = sitk_image[0, ...]

    h, w = image_single.shape
    ratio = 1024 / max(h, w)
    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized_img = misc.imresize(image_single, (new_h, new_w))

    cur_max = wc + (ww + 0.0) / 2
    cur_min = wc - (ww + 0.0) / 2

    norm_image = (resized_img - cur_min) / float(cur_max - cur_min)
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 1] = 1
    norm_image = norm_image * 255

    resized_img = np.array(norm_image, dtype=np.uint8)

    out_img = np.zeros([scale_size, scale_size], dtype=np.uint8)

    h_start = scale_size // 2 - new_h // 2
    w_start = scale_size // 2 - new_w // 2
    out_img[h_start:h_start + new_h, w_start:w_start + new_w] = resized_img

    basename = os.path.basename(infile).split('.')[0]
    outfile = os.path.join(out_dir, '{}.jpg'.format(basename))
    misc.imsave(outfile, out_img)
    print('====> save resize image:\t{}'.format(outfile))

def resize_dr_by_folder(infiles, scale_size, out_dir):
    for infile in infiles:
        resize_dr_dicom_to_size(infile, scale_size, out_dir)
    return 1

def batch_resize_dr_dicom_to_size(infiles, outdir, scale_size, process_num=2):
    num_per_process = (len(infiles) + process_num - 1) // process_num
    data_to_processed = []
    for i in range(process_num):
        data_to_processed.append(infiles[i*num_per_process:min((i+1)*num_per_process, len(infiles))])
    results = []
    pool = multiprocessing.Pool()
    for i in range(process_num):
        result = pool.apply_async(resize_dr_by_folder,
                                  args=(data_to_processed[i], scale_size, outdir))
        results.append(result)
    pool.close()
    pool.join()

def generate_data(config_file, outdir):
    print('====> begin test_batch_resize_dr_dicom_to_size:')
    config_file = config_file
    infiles = []
    with open(config_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            infiles.append(line)
    outdir = outdir
    batch_resize_dr_dicom_to_size(infiles, outdir, 1024, 8)
    print('====> end test_batch_resize_dr_dicom_to_size!')

def test_generate_data():
    config_file = []
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_pos_train.txt')
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_pos_val.txt')
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_pos_test.txt')
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_neg_train.txt')
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_neg_val.txt')
    config_file.append('/home/zhangwd/code/work/general_dr_cls/ccyy_dicom/total_neg_test.txt')
    outdir_root = '/hdd/disk3/wd/dr'
    for i in range(len(config_file)):
        basename = os.path.basename(config_file[i]).split('.')[0]
        outdir = os.path.join(outdir_root, basename)
        os.makedirs(outdir, exist_ok=True)
        generate_data(config_file[i], outdir)

if __name__ == '__main__':
    test_generate_data()