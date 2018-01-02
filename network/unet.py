import numpy as np
import tensorflow as tf
from network import network


class UnetModel(network.Network):
    def __init__(self, inputs, trainable, model_name, input_size, dropout_rate=0.2):
        network.Network.__init__(self, inputs, trainable, model_name, dropout_rate)
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')

    def load_weights(self, ckpt_dir, layers2load):
        layers_list = []
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        #tf.train.init_from_checkpoint(ckpt_dir, load_dict)
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

    def make_learning_rate(self, lr, decay_steps, decay_rate):
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)

    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            # pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.class_num])
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self, lr):
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step)

    def make_summary(self):
        tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train(self, x_name, y_name, epoch_num, n_train, batch_size, sess, summary_writer, n_valid=1000,
              train_iterator=None, train_reader=None, valid_iterator=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        for epoch in range(epoch_num):
            for step in range(0, n_train, batch_size):
                if train_iterator is not None:
                    X_batch, y_batch = next(train_iterator)
                else:
                    X_batch, y_batch = sess.run(train_reader)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            for step in range(0, n_valid, batch_size):
                if valid_iterator is not None:
                    X_batch_val, y_batch_val = next(valid_iterator)
                else:
                    X_batch_val, y_batch_val = sess.run(valid_reader)
                pred_valid, cross_entropy_valid = sess.run([self.pred, self.loss],
                                                           feed_dict={self.inputs[x_name]: X_batch_val,
                                                                      self.inputs[y_name]: y_batch_val,
                                                                      self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            print('Validation cross entropy: {:.3f}'.format(cross_entropy_valid_mean))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


    def test(self, x_name, sess, test_iterator, soft_pred=False):
        result = []
        for X_batch in test_iterator:
            if soft_pred:
                pred = sess.run(tf.nn.softmax(self.pred), feed_dict={self.inputs[x_name]:X_batch,
                                                                     self.trainable: False})
            else:
                pred = sess.run(self.pred, feed_dict={self.inputs[x_name]:X_batch,
                                                      self.trainable: False})
            result.append(pred)
        result = np.vstack(result)
        return result


class UnetModel_Origin(UnetModel):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')

    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            #pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.class_num])
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-184, h-184)
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))


class UnetModel_Height(UnetModel_Origin):
    def load_weights(self, ckpt_dir, layers2load, conv1_weight, check_weight=False):
        layers_list = []
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id == 1:
                layers_list.append('layerconv1/conv_1/bias:0')
                layers_list.append('layerconv1/bn_1')
                layers_list.append('layerconv1/conv_2')
                layers_list.append('layerconv1/bn_2')
                continue
            elif layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        #tf.train.init_from_checkpoint(ckpt_dir, load_dict)
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

        layerconv1_kernel = tf.trainable_variables()[0]
        assign_op = layerconv1_kernel.assign(conv1_weight)
        with tf.Session() as sess:
            sess.run(assign_op)
            weight = sess.run(layerconv1_kernel)

        if check_weight:
            import matplotlib.pyplot as plt
            _, _, c_num, _ = weight.shape
            for i in range(c_num):
                plt.subplot(321+i)
                plt.imshow(weight[:, :, i, :].reshape((16, 18)))
                plt.colorbar()
                plt.title(i)
            plt.show()


class UnetModel_Height_Appendix(UnetModel_Origin):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        conv10 = tf.layers.conv2d(conv9, sfn, (1, 1), name='second_final', padding='same')
        self.pred = tf.layers.conv2d(conv10, class_num, (1, 1), name='final', activation=None, padding='same')

    def load_weights(self, ckpt_dir, layers2load, conv1_weight, check_weight=False):
        layers_list = []
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id == 1:
                layers_list.append('layerconv1/conv_1/bias:0')
                layers_list.append('layerconv1/bn_1')
                layers_list.append('layerconv1/conv_2')
                layers_list.append('layerconv1/bn_2')
                continue
            elif layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        #tf.train.init_from_checkpoint(ckpt_dir, load_dict)
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

        layerconv1_kernel = tf.trainable_variables()[0]
        assign_op = layerconv1_kernel.assign(conv1_weight)
        with tf.Session() as sess:
            sess.run(assign_op)
            weight = sess.run(layerconv1_kernel)

        if check_weight:
            import matplotlib.pyplot as plt
            _, _, c_num, _ = weight.shape
            for i in range(c_num):
                plt.subplot(321+i)
                plt.imshow(weight[:, :, i, :].reshape((16, 18)))
                plt.colorbar()
                plt.title(i)
            plt.show()


class UnetModel_Height_Appendix_Weight(UnetModel_Height_Appendix):
    def make_loss(self, y_name, w_name):
        with tf.variable_scope('loss'):
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            z = self.inputs[w_name]
            z = tf.image.resize_image_with_crop_or_pad(z, w - 184, h - 184)
            z = tf.squeeze(z, axis=3)
            pred_flat = tf.reshape(tf.nn.softmax(tf.multiply(self.pred, tf.stack([z, z], axis=3))), [-1, self.class_num])
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w - 184, h - 184)
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                               labels=gt))
            '''self.loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=prediction,
                                                         targets=gt,
                                                         pos_weight=tf.constant([0.5, 1])))'''

    def train(self, x_name, y_name, z_name, epoch_num, n_train, batch_size, sess, summary_writer,
              train_iterator=None, train_reader=None, valid_iterator=None, valid_reader=None,
              image_summary=None, verb_step=100):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        for epoch in range(epoch_num):
            for step in range(0, n_train, batch_size):
                if train_iterator is not None:
                    X_batch, z_batch, y_batch = next(train_iterator)
                else:
                    X_batch, z_batch, y_batch = sess.run(train_reader)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.inputs[z_name]:z_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.inputs[z_name]: z_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            if valid_iterator is not None:
                X_batch_val, z_batch_val, y_batch_val = next(valid_iterator)
            else:
                X_batch_val, z_batch_val, y_batch_val = sess.run(valid_reader)
            pred_valid, cross_entropy_valid = sess.run([self.pred, self.loss],
                                                       feed_dict={self.inputs[x_name]: X_batch_val,
                                                                  self.inputs[y_name]: y_batch_val,
                                                                  self.inputs[z_name]: z_batch_val,
                                                                  self.trainable: False})
            print('Validation cross entropy: {:.3f}'.format(cross_entropy_valid))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)

            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
            saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)


class ResUnetModel(UnetModel):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_identity_pool(self.inputs[x_name], [sfn, sfn, sfn], self.trainable, name='conv1')
        conv2, pool2 = self.conv_conv_identity_pool(pool1, [sfn*2, sfn*2, sfn*2], self.trainable, name='conv2')
        conv3, pool3 = self.conv_conv_identity_pool(pool2, [sfn*4, sfn*4, sfn*4], self.trainable, name='conv3')
        conv4, pool4 = self.conv_conv_identity_pool(pool3, [sfn*8, sfn*8, sfn*8], self.trainable, name='conv4')
        conv5 = self.conv_conv_identity_pool(pool4, [sfn*16, sfn*16, sfn*16], self.trainable, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_identity_pool(up6, [sfn*8, sfn*8, sfn*8], self.trainable, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_identity_pool(up7, [sfn*4, sfn*4, sfn*4], self.trainable, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_identity_pool(up8, [sfn*2, sfn*2, sfn*2], self.trainable, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_identity_pool(up9, [sfn, sfn, sfn], self.trainable, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')


class ResUnetModel_Crop(UnetModel_Origin):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_identity_pool_crop(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                                         padding='valid')
        conv2, pool2 = self.conv_conv_identity_pool_crop(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                                         padding='valid')
        conv3, pool3 = self.conv_conv_identity_pool_crop(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                                         padding='valid')
        conv4, pool4 = self.conv_conv_identity_pool_crop(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                                         padding='valid')
        conv5 = self.conv_conv_identity_pool_crop(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                                  padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_identity_pool_crop(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                                  padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_identity_pool_crop(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                                  padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_identity_pool_crop(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                                  padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_identity_pool_crop(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                                  padding='valid')

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')


class ResUnetModel_shrink(UnetModel):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_identity_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1')
        conv2, pool2 = self.conv_conv_identity_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2')
        conv3, pool3 = self.conv_conv_identity_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3')
        conv4, pool4 = self.conv_conv_identity_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4')
        conv5 = self.conv_conv_identity_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_identity_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_identity_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_identity_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_identity_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')


class ResUnetSkipModel(UnetModel):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1 = self.conv_conv_identity_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', pool=False)
        conv1_skip, pool1 = self.conv_conv_skip_pool(conv1, [sfn, sfn], self.trainable, name='conv1_skip')
        conv2 = self.conv_conv_identity_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', pool=False)
        conv2_skip, pool2 = self.conv_conv_skip_pool(conv2, [sfn*2, sfn*2], self.trainable, name='conv2_skip')
        conv3 = self.conv_conv_identity_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', pool=False)
        conv3_skip, pool3 = self.conv_conv_skip_pool(conv3, [sfn*4, sfn*4], self.trainable, name='conv3_skip')
        conv4 = self.conv_conv_identity_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', pool=False)
        conv4_skip, pool4 = self.conv_conv_skip_pool(conv4, [sfn*8, sfn*8], self.trainable, name='conv4_skip')
        conv5 = self.conv_conv_identity_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False)
        conv5 = self.conv_conv_skip_pool(conv5, [sfn*16, sfn*16], self.trainable, name='conv5_skip', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4_skip, name='6')
        conv6 = self.conv_conv_identity_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False)
        conv6_skip = self.conv_conv_skip_pool(conv6, [sfn*8, sfn*8], self.trainable, name='up6_skip', pool=False)
        up7 = self.upsample_concat(conv6_skip, conv3_skip, name='7')
        conv7 = self.conv_conv_identity_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False)
        conv7_skip = self.conv_conv_skip_pool(conv7, [sfn*4, sfn*4], self.trainable, name='up7_skip', pool=False)
        up8 = self.upsample_concat(conv7_skip, conv2_skip, name='8')
        conv8 = self.conv_conv_identity_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False)
        conv8_skip = self.conv_conv_skip_pool(conv8, [sfn*2, sfn*2], self.trainable, name='up8_skip', pool=False)
        up9 = self.upsample_concat(conv8_skip, conv1_skip, name='9')
        conv9 = self.conv_conv_identity_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False)
        conv9_skip = self.conv_conv_skip_pool(conv9, [sfn, sfn], self.trainable, name='up9_skip', pool=False)

        self.pred = tf.layers.conv2d(conv9_skip, class_num, (1, 1), name='final', activation=None, padding='same')
