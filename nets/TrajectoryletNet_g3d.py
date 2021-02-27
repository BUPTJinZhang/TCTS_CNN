import tensorflow as tf
from layers.Trajectoryblock import TrajBlock as TB
from layers.Trajectoryblock import TrajBlock_2str as TB_str
from layers.Trajectoryblock_g3d import TrajBlock as TB_g3d
from layers.Trajectoryblock_g3d import TrajBlock_2str as TB_g3d_str
import pdb
import numpy as np


def TrajNet(images, keep_prob, seq_length, input_length, stacklength, num_hidden, filter_size):
    with tf.variable_scope('TrajNet', reuse=False):
        print('TrajectoryletNet_final')
        # print 'is_training', is_training
        h = images[:, 0:seq_length, :, :]
        gt_images = images[:, seq_length:]
        dims = gt_images.shape[-1]
        inputs = h
        inputs = tf.transpose(h, [0, 2, 3, 1])

        out = []
        loss = 0
        inputs = tf.layers.conv2d(inputs, num_hidden[0], 1, padding='same', activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='h0')
        # p = inputs.numpy().copy()
        for i in range(stacklength):
            inputs = TB('TrajBlock' + str(i), filter_size, num_hidden, keep_prob)(inputs)

        out = tf.layers.conv2d(inputs, seq_length - input_length, filter_size, padding='same',
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv')

        out = tf.layers.conv2d(out, seq_length - input_length, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv1')

        # pdb.set_tracie()
        out = tf.transpose(out, [0, 3, 1, 2])

        # loss
        gen_images = out
        loss += tf.reduce_mean(tf.norm(gen_images - gt_images, axis=3, keep_dims=True, name='normal'))
        return [gen_images, loss]


def TrajNet_2str(images, images_v, keep_prob, seq_length, input_length, stacklength, num_hidden, filter_size):
    with tf.variable_scope('TrajNet', reuse=False):
        print('TrajectoryletNet_final')
        # print 'is_training', is_training
        # =================POSITION PROCESSING======================
        h = images[:, 0:seq_length, :, :]
        gt_images = images[:, seq_length:]
        dims = gt_images.shape[-1]
        inputs = h
        inputs = tf.transpose(h, [0, 2, 3, 1])
        # ==========================================================
        # =================VELOCITY PROCESSING======================
        h_v = images_v[:, 0:seq_length, :, :]
        # gt_images_v = images_v[:, seq_length:]
        # dims = gt_images.shape[-1]
        inputs_v = h_v
        inputs_v = tf.transpose(h_v, [0, 2, 3, 1])
        # ==========================================================

        out = []
        loss = 0
        inputs = tf.layers.conv2d(inputs, num_hidden[0], 1, padding='same', activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='h0')
        inputs_v = tf.layers.conv2d(inputs_v, num_hidden[0], 1, padding='same', activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='h0_v')
        # p = inputs.numpy().copy()
        for i in range(2):
            inputs = TB_g3d('TrajBlock' + str(i), filter_size, num_hidden, keep_prob)(inputs)
        for i in range(2):
            inputs_v = TB_g3d_str('TrajBlock' + str(i), filter_size, num_hidden, keep_prob)(inputs_v)

        out = tf.layers.conv2d(inputs, seq_length - input_length, filter_size, padding='same',
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv')

        out = tf.layers.conv2d(out, seq_length - input_length, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv1')
        # ====================================================================================
        out_v = tf.layers.conv2d(inputs_v, seq_length - input_length, filter_size, padding='same',
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv_v')

        out_v = tf.layers.conv2d(out_v, seq_length - input_length, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='trajout_conv1_v')
        # ==============================================================================
        out_fromv = tf.zeros(out_v.shape)
        tem_0 = inputs[:, :, :, 9] + out_v[:, :, :, 0]
        tem_1 = tem_0 + out_v[:, :, :, 1]
        tem_2 = tem_1 + out_v[:, :, :, 2]
        tem_3 = tem_2 + out_v[:, :, :, 3]
        tem_4 = tem_3 + out_v[:, :, :, 4]
        tem_5 = tem_4 + out_v[:, :, :, 5]
        tem_6 = tem_5 + out_v[:, :, :, 6]
        tem_7 = tem_6 + out_v[:, :, :, 7]
        tem_8 = tem_7 + out_v[:, :, :, 8]
        tem_9 = tem_8 + out_v[:, :, :, 9]
        out_fromv = tf.stack([tem_0, tem_1, tem_2, tem_3, tem_4, tem_5, tem_6, tem_7, tem_8, tem_9], axis=3)
        # =====================================================
        tem_p0 = out[:, :, :, 0]
        tem_v0 = out_fromv[:, :, :, 0]
        tem_p0 = tf.expand_dims(tem_p0, axis=3)
        tem_v0 = tf.expand_dims(tem_v0, axis=3)
        y_0 = tf.concat([tem_p0, tem_v0], axis=3)
        y_0 = tf.layers.conv2d(y_0, 1, 1, padding='same', activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='concat0')

        tem_p1 = out[:, :, :, 1]
        tem_v1 = out_fromv[:, :, :, 1]
        tem_p1 = tf.expand_dims(tem_p1, axis=3)
        tem_v1 = tf.expand_dims(tem_v1, axis=3)
        y_1 = tf.concat([tem_p1, tem_v1], axis=3)
        y_1 = tf.layers.conv2d(y_1, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat1')

        tem_p2 = out[:, :, :, 2]
        tem_v2 = out_fromv[:, :, :, 2]
        tem_p2 = tf.expand_dims(tem_p2, axis=3)
        tem_v2 = tf.expand_dims(tem_v2, axis=3)
        y_2 = tf.concat([tem_p2, tem_v2], axis=3)
        y_2 = tf.layers.conv2d(y_2, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat2')

        tem_p3 = out[:, :, :, 3]
        tem_v3 = out_fromv[:, :, :, 3]
        tem_p3 = tf.expand_dims(tem_p3, axis=3)
        tem_v3 = tf.expand_dims(tem_v3, axis=3)
        y_3 = tf.concat([tem_p3, tem_v3], axis=3)
        y_3 = tf.layers.conv2d(y_3, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat3')

        tem_p4 = out[:, :, :, 4]
        tem_v4 = out_fromv[:, :, :, 4]
        tem_p4 = tf.expand_dims(tem_p4, axis=3)
        tem_v4 = tf.expand_dims(tem_v4, axis=3)
        y_4 = tf.concat([tem_p4, tem_v4], axis=3)
        y_4 = tf.layers.conv2d(y_4, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat4')

        tem_p5 = out[:, :, :, 5]
        tem_v5 = out_fromv[:, :, :, 5]
        tem_p5 = tf.expand_dims(tem_p5, axis=3)
        tem_v5 = tf.expand_dims(tem_v5, axis=3)
        y_5 = tf.concat([tem_p5, tem_v5], axis=3)
        y_5 = tf.layers.conv2d(y_5, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat5')

        tem_p6 = out[:, :, :, 6]
        tem_v6 = out_fromv[:, :, :, 6]
        tem_p6 = tf.expand_dims(tem_p6, axis=3)
        tem_v6 = tf.expand_dims(tem_v6, axis=3)
        y_6 = tf.concat([tem_p6, tem_v6], axis=3)
        y_6 = tf.layers.conv2d(y_6, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat6')

        tem_p7 = out[:, :, :, 7]
        tem_v7 = out_fromv[:, :, :, 7]
        tem_p7 = tf.expand_dims(tem_p7, axis=3)
        tem_v7 = tf.expand_dims(tem_v7, axis=3)
        y_7 = tf.concat([tem_p7, tem_v7], axis=3)
        y_7 = tf.layers.conv2d(y_7, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat7')

        tem_p8 = out[:, :, :, 8]
        tem_v8 = out_fromv[:, :, :, 8]
        tem_p8 = tf.expand_dims(tem_p8, axis=3)
        tem_v8 = tf.expand_dims(tem_v8, axis=3)
        y_8 = tf.concat([tem_p8, tem_v8], axis=3)
        y_8 = tf.layers.conv2d(y_8, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat8')

        tem_p9 = out[:, :, :, 9]
        tem_v9 = out_fromv[:, :, :, 9]
        tem_p9 = tf.expand_dims(tem_p9, axis=3)
        tem_v9 = tf.expand_dims(tem_v9, axis=3)
        y_9 = tf.concat([tem_p9, tem_v9], axis=3)
        y_9 = tf.layers.conv2d(y_9, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat9')
        # for j in range(seq_length - input_length):
        #     tem = out[:, :, :, j]
        #     tem_v = out_fromv[:, :, :, j]
        #     tem = tf.expand_dims(tem, axis=3)
        #     tem_v = tf.expand_dims(tem_v, axis=3)
        #     y_0 = tf.concat([tem, tem_v], axis=3)
        #     if j == 0:
        #         z = y_0
        #     else:
        z = tf.concat([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9], axis=3)
        z = TB_g3d('TrajBlockout', filter_size, num_hidden, keep_prob)(z)
        z = TB_g3d('TrajBlockout_1', filter_size, num_hidden, keep_prob)(z)
        # z = TB('TrajBlockout_2', filter_size, num_hidden, keep_prob)(z)
        z = tf.layers.conv2d(z, seq_length - input_length, filter_size, padding='same',
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='2strout_conv')

        z = tf.layers.conv2d(z, seq_length - input_length, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='2strout_conv1')




        # pdb.set_tracie()
        z = tf.transpose(z, [0, 3, 1, 2])
        # out = tf.transpose(out, [0, 3, 1, 2])

        # loss
        gen_images = z
        loss += tf.reduce_mean(tf.norm(gen_images - gt_images, axis=3, keep_dims=True, name='normal'))
        return [gen_images, loss]
