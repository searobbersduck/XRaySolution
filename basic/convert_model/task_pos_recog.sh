# dr position recognition
dr_name="posrecog"
python 1.convert_pytorch_model_to_onnx.py --weights ../../dr_pos_recog/train/model/dr_cls_pos_neg/ct_pos_recognition_0023_best.pth
CUDA_VISIBLE_DEVICES=0 python 2.convert_onnx_model_to_tf.py
rm -rf export_${dr_name}
CUDA_VISIBLE_DEVICES=0 python 3.convert_pb_to_signature_pb.py --outdir export_${dr_name}
rm -rf $1/${dr_name}_cls
cp -r $1/dr_cls $1/${dr_name}_cls
sed -i 's|"dr_cls"|"posrecog_cls"|' $1/${dr_name}_cls/config.pbtxt
cp -r ./export_${dr_name}/* $1/${dr_name}_cls/1/model.savedmodel/

