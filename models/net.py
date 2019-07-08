from math import log

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_normalizer(is_training):
    bn_fn_args = {
        'is_training': is_training,
        'fused': True,
        'scale': True,
        'decay': 0.975,
        'epsilon': 0.001,
    }
    return slim.batch_norm, bn_fn_args


def discriminator(inputs,
                  depth=64,
                  is_training=True,
                  reuse=None,
                  scope='discriminator'):
    normalizer_fn, normalizer_fn_args = get_normalizer(is_training)
    inp_shape = inputs.get_shape().as_list()[1]
    act_fn = tf.nn.leaky_relu
    print('Activation function: {}'.format(act_fn))

    end_points = {}
    with tf.variable_scope(scope, values=[inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            stride=2,
                            kernel_size=4,
                            activation_fn=act_fn):
            net = inputs
            net = slim.conv2d(net, depth, kernel_size=3, stride=1, normalizer_fn=None, scope='conv0')
            for i in xrange(int(log(inp_shape, 2))-2):
                scope = 'conv%i' % (i + 1)
                current_depth = min(depth * 2 ** (i+1), 1024)
                net = slim.conv2d(net, current_depth, normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_fn_args, scope=scope)
                print('Discriminator layer {}: {}'.format(i, net.get_shape().as_list()))
                end_points[scope] = net

            last_shape = net.get_shape().as_list()
            logits = slim.conv2d(net, 1, kernel_size=last_shape[1], stride=1, padding='VALID',
                                 normalizer_fn=None, activation_fn=None)
            logits = tf.reshape(logits, [-1, 1])
            end_points['logits'] = logits

            return logits, end_points


def generator(inputs,
              depth=64,
              final_size=32,
              num_outputs=3,
              is_training=True,
              reuse=None,
              scope='generator'):
    normalizer_fn, normalizer_fn_args = get_normalizer(is_training)
    act_fn = tf.nn.relu

    inputs.get_shape().assert_has_rank(2)

    end_points = {}
    num_layers = int(log(final_size, 2)) - 1
    with tf.variable_scope(scope, values=[inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose],
                            normalizer_fn=normalizer_fn,
                            stride=2,
                            kernel_size=4,
                            activation_fn=act_fn):
            net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

            # First upscaling is different because it takes the input vector.
            current_depth = min(depth * 2 ** (num_layers - 1), 1024)
            scope = 'deconv1'
            net = slim.conv2d_transpose(net, current_depth, normalizer_fn=normalizer_fn,
                                        kernel_size=final_size//(2**(num_layers-1)),
                                        normalizer_params=normalizer_fn_args, stride=1,
                                        padding='VALID', scope=scope)
            end_points[scope] = net
            print('Generator layer {}: {}'.format(1, net.get_shape().as_list()))

            for i in xrange(2, num_layers + 1):
                scope = 'deconv%i' % (i)
                current_depth = min(depth * 2 ** (num_layers - i), 1024)
                net = slim.conv2d_transpose(net, current_depth,  normalizer_fn=normalizer_fn,
                                            normalizer_params=normalizer_fn_args, scope=scope)
                print('Generator layer {}: {}'.format(i, net.get_shape().as_list()))
                end_points[scope] = net

            # Last layer has different normalizer and activation.
            scope = 'deconv%i' % (num_layers+1)
            net = slim.conv2d_transpose(net, num_outputs, kernel_size=3, stride=1,
                                        normalizer_fn=None, activation_fn=None, scope=scope)
            end_points[scope] = net

            print('Generator output: {}'.format(net.get_shape().as_list()))

            return net, end_points

