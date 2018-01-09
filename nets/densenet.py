"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # Put your code here.
            #Convolution 2k convolutions of size 7×7 with stride 2
            net=slim.conv2d(images, 2*growth, [7,7],padding='SAME', scope='Conv2d_0a_7x7')
            end_points['Conv2d_0a_7x7'] = net

            #Pooling the number of feature-maps in all other layers also follow from setting k
            #Pooling out size 56x56,kernel size[3,3] stride 2,method pooling:max
            #Pooling Layer 
            net=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='MaxPool_0a_3x3')
            end_points['MaxPool_0a_3x3'] = net

            #Dense Block 1   
            #layers:6
            net=block(net,6,growth,'Dense_Block_1')
            end_points['Dense_Block_1'] = net

            #Transition Layer 1
            output_num_1=reduce_dim(net)
            net=slim.conv2d(net,output_num_1,[1,1],padding='SAME',scope='Conv2d_1a_1x1')
            end_points['Conv2d_1a_1x1'] = net
            net=slim.avg_pool2d(net,[2,2],stride=2,padding='VALID',scope='AvgPool_1a_2x2')
            end_points['AvgPool_1a_2x2'] = net

            #Dense Block 2   
            #layers:12
            net=block(net,12,growth,'Dense_Block_2')
            end_points['Dense_Block_2'] = net

            #Transition Layer 2
            output_num_2=reduce_dim(net)
            net=slim.conv2d(net,output_num_2,[1,1],padding='SAME',scope='Conv2d_2a_1x1')
            end_points['Conv2d_2a_1x1'] = net
            net=slim.avg_pool2d(net,[2,2],stride=2,padding='VALID',scope='AvgPool_2a_2x2')
            end_points['AvgPool_2a_2x2'] = net

            #Dense Block 3   
            #layers:32
            net=block(net,32,growth,'Dense_Block_3')
            end_points['Dense_Block_3'] = net

            #Transition Layer 3
            output_num_3=reduce_dim(net)
            net=slim.conv2d(net,output_num_3,[1,1],padding='SAME',scope='Conv2d_3a_1x1')
            end_points['Conv2d_3a_1x1'] = net
            net=slim.avg_pool2d(net,[2,2],stride=2,padding='VALID',scope='AvgPool_3a_2x2')
            end_points['AvgPool_3a_2x2'] = net

            #Dense Block 4
            #layers:32
            net=block(net,32,growth,'Dense_Block_4')
            end_points['Dense_Block_4'] = net

            with tf.variable_scope('Logits'):
                #7x7x128
                net=slim.avg_pool2d(net,[7,7],padding='VALID',scope='AvgPool_3a_7x7')
                end_points['AvgPool_4a_7x7'] = net
                logits = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits')
                end_points['Logits'] = Logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
