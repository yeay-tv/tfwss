"""
net_surgery.py

VGG16 Transfer Learning After 3-to-4-Channel Input Conversion

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/minhnhat93/tf_object_detection_multi_channels/blob/master/edit_checkpoint.py
    Written by SNhat M. Nguyen
    Unknown code license
"""
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf

num_input_channels = 4 # AStream uses 4-channel inputs
init_method = 'gaussian' # ['gaussian'|'spread_average'|'zeros']
input_path = 'models/vgg_16_3chan/vgg_16_3chan.ckpt' # copy of checkpoint in http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
output_path = 'models/vgg_16_4chan/vgg_16_4chan.ckpt'

print('Loading checkpoint...')
reader = pywrap_tensorflow.NewCheckpointReader(input_path)
print('...done loading checkpoint.')

var_to_shape_map = reader.get_variable_to_shape_map()
var_to_edit_name = 'vgg_16/conv1/conv1_1/weights'

for key in sorted(var_to_shape_map):
    if key != var_to_edit_name:
        var = tf.Variable(reader.get_tensor(key), name=key, dtype=tf.float32)
    else:
        var_to_edit = reader.get_tensor(var_to_edit_name)
        print('Tensor {} of shape {} located.'.format(var_to_edit_name, var_to_edit.shape))

sess = tf.Session()
if init_method != 'gaussian':
    print('Error: Unimplemented initialization method')
new_channels_shape = list(var_to_edit.shape)
new_channels_shape[2] = num_input_channels - 3
gaussian_var = tf.random_normal(shape=new_channels_shape, stddev=0.001).eval(session=sess)
new_var = np.concatenate([var_to_edit, gaussian_var], axis=2)
new_var = tf.Variable(new_var, name=var_to_edit_name, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, output_path)


