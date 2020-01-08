# DR疾病分类
## 配置文件
```
./config/config_dr_dr.json
./config/config_dr_zhudongmai.json
./config/config_dr_zhanwei.json
./config/config_dr_shibian.json
./config/config_dr_xinying.json
./config/config_dr_shuhou.json
./config/config_dr_qixiong.json
./config/config_dr_yanxing.json
./config/config_dr_xianwei.json
./config/config_dr_wenli.json
./config/config_dr_feibuzhang.json
./config/config_dr_feidapao.json
./config/config_dr_feiqizhong.json
./config/config_dr_xiongqiangjiye.json
```
## 训练
```
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_dr.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_zhudongmai.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_zhanwei.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_shibian.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_xinying.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_shuhou.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_qixiong.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_yanxing.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_xianwei.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_wenli.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_feibuzhang.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_feidapao.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_feiqizhong.json
CUDA_VISIBLE_DEVICES=3,4,5,7 python train_dr.py --config_file ./config/config_dr_xiongqiangjiye.json

```

## 测试
[测试](./predict_dr.ipynb)

## reference

