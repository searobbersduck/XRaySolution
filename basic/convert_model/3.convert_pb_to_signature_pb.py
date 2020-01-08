import os
import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='convert pytorch model to onnx')
    parser.add_argument('--inname', default='dr.onnx')
    parser.add_argument('--outdir', default='./export')
    return parser.parse_args()

opt = parse_args()

infile = 'dr.pb'

graph = tf.get_default_graph()
sess = tf.Session()

with open(infile, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

inp = sess.graph.get_tensor_by_name('input:0')
print('====> input:\t')
print(inp)
outp = sess.graph.get_tensor_by_name('add_17:0')
print('====> output:\t')
print(outp)

# export_path = './export'
export_path = opt.outdir
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_input = tf.saved_model.utils.build_tensor_info(inp)
tensor_info_output = tf.saved_model.utils.build_tensor_info(outp)

#定义签名
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'images': tensor_info_input},
    outputs={'result': tensor_info_output},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

# 'serving_default' 可以随意定义，但是直接利用系统的trt框架挂载，默认要填'serving_default'
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={'serving_default': prediction_signature})
builder.save()
print('Done exporting!')
