# feibuzhang
dr_name="feibuzhang"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0000_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"feibuzhang_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="feidapao"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0030_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"feidapao_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="feiqizhong"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0035_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"feiqizhong_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="qixiong"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0035_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"qixiong_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="shibian"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0000_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"shibian_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="shuhou"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0039_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"shuhou_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="wenli"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0033_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"wenli_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="xianwei"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0025_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"xianwei_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="xiongqiangjiye"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0039_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"xiongqiangjiye_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="yanxing"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0034_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"yanxing_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="zhanwei"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0031_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"zhanwei_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/


dr_name="zhudongmai"
python 1.convert_pytorch_model_to_onnx.py --weights ../train/model/dr_cls_${dr_name}/ct_pos_recognition_0010_best.pth
CUDA_VISIBLE_DEVICES=2 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=2 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"zhudongmai_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/
