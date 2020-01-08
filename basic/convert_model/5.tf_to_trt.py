from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf

infile = 'dr.pb'

graph = tf.get_default_graph()
sess = tf.Session()

with open(infile, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

print('111111111111111')
converter = trt.TrtGraphConverter(input_graph_def=graph_def, precision_mode='FP16')
print('222222222222222')
converter.convert()
print('3333333333333333')
converter.save('./trt_yyy')
