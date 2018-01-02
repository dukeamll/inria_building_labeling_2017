import os
import tensorflow as tf


class Network(object):
    def __init__(self, inputs, trainable, model_name, dropout_rate=0.2):
        self.inputs = inputs
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.global_step_value = 0
        self.class_num = []
        self.loss = []
        self.optimizer = []
        self.pred = []
        self.summary = []
        self.ckdir = []

    def create_graph(self, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

    def make_ckdir(self, ckdir):
        self.ckdir = os.path.join(ckdir, self.model_name)

    def load(self, model_path):
        # TODO load pretrained model from path
        pass

    def get_unique_name(self, prefix):
        # TODO give layer a unique name with prefix
        pass

    def conv_conv_pool(self, input_, n_filters, training, name, conv_strid=(3, 3),
                       pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                       activation=tf.nn.relu, padding='same', bn=True):
        net = input_

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, conv_strid, activation=None,
                                       padding=padding, name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def conv_conv_identity_pool(self, input_, n_filters, training, name, conv_strid=(3, 3),
                                pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                                activation=tf.nn.relu, bn=True):
        net = input_

        with tf.variable_scope('layer{}'.format(name)):
            input_conv = tf.layers.conv2d(net, n_filters[-1], conv_strid, activation=None,
                                          padding='same', name='conv_skip')

            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, conv_strid, activation=None,
                                       padding='same', name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))

            # identity connection
            net = tf.add(input_conv, net)
            net = activation(net, name='relu_{}'.format(name, len(n_filters) + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def conv_conv_identity_pool_crop(self, input_, n_filters, training, name, conv_strid=(3, 3),
                                     pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                                     activation=tf.nn.relu, padding='same', bn=True):
        net = input_
        _, w, h, _ = input_.get_shape().as_list()
        with tf.variable_scope('layer{}'.format(name)):
            input_conv = tf.layers.conv2d(net, n_filters[-1], conv_strid, activation=None,
                                          padding='same', name='conv_skip')
            input_conv = tf.layers.batch_normalization(input_conv, training=training, name='bn_skip')

            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, conv_strid, activation=None,
                                       padding=padding, name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))

            # identity connection
            if padding == 'valid':
                input_conv = tf.image.resize_image_with_crop_or_pad(input_conv, w-2*len(n_filters), h-2*len(n_filters))
            net = tf.add(input_conv, net)
            net = activation(net, name='relu_{}'.format(name, len(n_filters) + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def conv_conv_skip_pool(self, input_, n_filters, training, name, conv_strid=(3, 3),
                                pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                                activation=tf.nn.relu, bn=True):
        net = input_

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, conv_strid, activation=None,
                                       padding='same', name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))

            # identity connection
            net = tf.add(input_, net)
            net = activation(net, name='relu_{}'.format(name, len(n_filters) + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def concat(self, input_a, input_b, training, name):
        with tf.variable_scope('layer{}'.format(name)):
            inputA_norm = tf.layers.batch_normalization(input_a, training=training, name='bn')
            return tf.concat([inputA_norm, input_b], axis=-1, name='concat_{}'.format(name))

    def upsampling_2D(self, tensor, name, size=(2, 2)):
        H, W, _ = tensor.get_shape().as_list()[1:]  # first dim is batch num
        H_multi, W_multi = size
        target_H = H * H_multi
        target_W = W * W_multi

        return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name='upsample_{}'.format(name))

    def upsample_concat(self, input_a, input_b, name):
        upsample = self.upsampling_2D(input_a, size=(2, 2), name=name)
        return tf.concat([upsample, input_b], axis=-1, name='concat_{}'.format(name))

    def crop_upsample_concat(self, input_a, input_b, margin, name):
        with tf.variable_scope('crop_upsample_concat'):
            _, w, h, _ = input_b.get_shape().as_list()
            input_b_crop = tf.image.resize_image_with_crop_or_pad(input_b, w-margin, h-margin)
            return self.upsample_concat(input_a, input_b_crop, name)

    def fc_fc(self, input_, n_filters, training, name, activation=tf.nn.relu, dropout=True):
        net = input_
        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.dense(net, F, activation=None)
                if activation is not None:
                    net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training, name='drop_{}'.format(name, i + 1))
        return net
