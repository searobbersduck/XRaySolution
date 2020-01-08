import os
from glob import glob

def generate_label_file(pos_dir, neg_dir, label_file):
    pos_files = glob(os.path.join(pos_dir, '*.jpg'))
    neg_files = glob(os.path.join(neg_dir, '*.jpg'))
    with open(label_file, 'w') as f:
        for pos_f in pos_files:
            f.write('{}\t{}\n'.format(pos_f, 1))
        for neg_f in neg_files:
            f.write('{}\t{}\n'.format(neg_f, 0))

def process_data(folder_name1,folder_name2):
    # folder_name1 = 'dr_deformable_1024'
    # folder_name2 = 'xxx'

    root_dir = '/data/zhangwd/data/xray/{}/{}'.format(folder_name1, folder_name2)
    train_pos_dir = '/data/zhangwd/data/xray/{}/{}/{}_pos_train/'.format(folder_name1, folder_name2, folder_name2)
    train_neg_dir = '/data/zhangwd/data/xray/{}/{}/{}_neg_train/'.format(folder_name1, folder_name2, folder_name2)

    val_pos_dir = '/data/zhangwd/data/xray/{}/{}/{}_pos_val/'.format(folder_name1, folder_name2, folder_name2)
    val_neg_dir = '/data/zhangwd/data/xray/{}/{}/{}_neg_val/'.format(folder_name1, folder_name2, folder_name2)

    test_pos_dir = '/data/zhangwd/data/xray/{}/{}/{}_pos_test/'.format(folder_name1, folder_name2, folder_name2)
    test_neg_dir = '/data/zhangwd/data/xray/{}/{}/{}_neg_test/'.format(folder_name1, folder_name2, folder_name2)

    train_label_file = os.path.join(root_dir, 'train_label.txt')
    val_label_file = os.path.join(root_dir, 'val_label.txt')
    test_label_file = os.path.join(root_dir, 'test_label.txt')

    generate_label_file(train_pos_dir, train_neg_dir, train_label_file)
    generate_label_file(val_pos_dir, val_neg_dir, val_label_file)
    generate_label_file(test_pos_dir, test_neg_dir, test_label_file)

def test_process_data():
    folder_name1 = 'dr_deformable_1024'
    cats = ['主动脉钙化', '占位', '实变', '心影增大', '术后', '气胸', '炎性改变', '纤维', '纹理增强', '肺不张', '肺大泡', '肺气肿', '胸腔积液']
    for cat in cats:
        process_data(folder_name1, cat)

if __name__ == '__main__':
    test_process_data()