import tensorflow as tf
from layers.Trajectoryblock import TrajBlock as TB
from layers.Trajectoryblock import TrajBlock_2str as TB_str
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

        inputs = TB('TrajBlock1', filter_size, num_hidden, keep_prob)(inputs)
        inputs = tf.layers.conv2d(inputs, num_hidden[0], filter_size, padding='same', activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='h1')
        inputs = TB('TrajBlock2', filter_size, num_hidden, keep_prob)(inputs)
        inputs_v = TB_str('TrajBlock1', filter_size, num_hidden, keep_prob)(inputs_v)
        inputs_v = tf.layers.conv2d(inputs_v, num_hidden[0], filter_size, padding='same', activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='h1_v')
        inputs_v = TB_str('TrajBlock2', filter_size, num_hidden, keep_prob)(inputs_v)

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
        tem_10 = tem_9 + out_v[:, :, :, 10]
        tem_11 = tem_10 + out_v[:, :, :, 11]
        tem_12 = tem_11 + out_v[:, :, :, 12]
        tem_13 = tem_12 + out_v[:, :, :, 13]
        tem_14 = tem_13 + out_v[:, :, :, 14]
        tem_15 = tem_14 + out_v[:, :, :, 15]
        tem_16 = tem_15 + out_v[:, :, :, 16]
        tem_17 = tem_16 + out_v[:, :, :, 17]
        tem_18 = tem_17 + out_v[:, :, :, 18]
        tem_19 = tem_18 + out_v[:, :, :, 19]
        tem_20 = tem_19 + out_v[:, :, :, 20]
        tem_21 = tem_20 + out_v[:, :, :, 21]
        tem_22 = tem_21 + out_v[:, :, :, 22]
        tem_23 = tem_22 + out_v[:, :, :, 23]
        tem_24 = tem_23 + out_v[:, :, :, 24]
        tem_25 = tem_24 + out_v[:, :, :, 25]
        tem_26 = tem_25 + out_v[:, :, :, 26]
        tem_27 = tem_26 + out_v[:, :, :, 27]
        tem_28 = tem_27 + out_v[:, :, :, 28]
        tem_29 = tem_28 + out_v[:, :, :, 19]
        out_fromv = tf.stack([tem_0, tem_1, tem_2, tem_3, tem_4, tem_5, tem_6, tem_7, tem_8, tem_9, tem_10,
                              tem_11, tem_12, tem_13, tem_14, tem_15, tem_16, tem_17, tem_18, tem_19,
                              tem_20, tem_21, tem_22, tem_23, tem_24, tem_25, tem_26, tem_27, tem_28, tem_29], axis=3)
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

        tem_p10 = out[:, :, :, 10]
        tem_v10 = out_fromv[:, :, :, 10]
        tem_p10 = tf.expand_dims(tem_p10, axis=3)
        tem_v10 = tf.expand_dims(tem_v10, axis=3)
        y_10 = tf.concat([tem_p10, tem_v10], axis=3)
        y_10 = tf.layers.conv2d(y_10, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat10')

        tem_p11 = out[:, :, :, 11]
        tem_v11 = out_fromv[:, :, :, 11]
        tem_p11 = tf.expand_dims(tem_p11, axis=3)
        tem_v11 = tf.expand_dims(tem_v11, axis=3)
        y_11 = tf.concat([tem_p11, tem_v11], axis=3)
        y_11 = tf.layers.conv2d(y_11, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat11')

        tem_p12 = out[:, :, :, 12]
        tem_v12 = out_fromv[:, :, :, 12]
        tem_p12 = tf.expand_dims(tem_p12, axis=3)
        tem_v12 = tf.expand_dims(tem_v12, axis=3)
        y_12 = tf.concat([tem_p12, tem_v12], axis=3)
        y_12 = tf.layers.conv2d(y_12, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat12')

        tem_p13 = out[:, :, :, 13]
        tem_v13 = out_fromv[:, :, :, 13]
        tem_p13 = tf.expand_dims(tem_p13, axis=3)
        tem_v13 = tf.expand_dims(tem_v13, axis=3)
        y_13 = tf.concat([tem_p13, tem_v13], axis=3)
        y_13 = tf.layers.conv2d(y_13, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat13')

        tem_p14 = out[:, :, :, 14]
        tem_v14 = out_fromv[:, :, :, 14]
        tem_p14 = tf.expand_dims(tem_p14, axis=3)
        tem_v14 = tf.expand_dims(tem_v14, axis=3)
        y_14 = tf.concat([tem_p14, tem_v14], axis=3)
        y_14 = tf.layers.conv2d(y_14, 1, 1, padding='same', activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='concat14')

        tem_p15 = out[:, :, :, 15]
        tem_v15 = out_fromv[:, :, :, 15]
        tem_p15 = tf.expand_dims(tem_p15, axis=3)
        tem_v15 = tf.expand_dims(tem_v15, axis=3)
        y_15 = tf.concat([tem_p15, tem_v15], axis=3)
        y_15 = tf.layers.conv2d(y_15, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat15')

        tem_p16 = out[:, :, :, 16]
        tem_v16 = out_fromv[:, :, :, 16]
        tem_p16 = tf.expand_dims(tem_p16, axis=3)
        tem_v16 = tf.expand_dims(tem_v16, axis=3)
        y_16 = tf.concat([tem_p16, tem_v16], axis=3)
        y_16 = tf.layers.conv2d(y_16, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat16')

        tem_p17 = out[:, :, :, 17]
        tem_v17 = out_fromv[:, :, :, 17]
        tem_p17 = tf.expand_dims(tem_p17, axis=3)
        tem_v17 = tf.expand_dims(tem_v17, axis=3)
        y_17 = tf.concat([tem_p17, tem_v17], axis=3)
        y_17 = tf.layers.conv2d(y_17, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat17')

        tem_p18 = out[:, :, :, 18]
        tem_v18 = out_fromv[:, :, :, 18]
        tem_p18 = tf.expand_dims(tem_p18, axis=3)
        tem_v18 = tf.expand_dims(tem_v18, axis=3)
        y_18 = tf.concat([tem_p18, tem_v18], axis=3)
        y_18 = tf.layers.conv2d(y_18, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat18')

        tem_p19 = out[:, :, :, 19]
        tem_v19 = out_fromv[:, :, :, 19]
        tem_p19 = tf.expand_dims(tem_p19, axis=3)
        tem_v19 = tf.expand_dims(tem_v19, axis=3)
        y_19 = tf.concat([tem_p17, tem_v19], axis=3)
        y_19 = tf.layers.conv2d(y_19, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat19')

        tem_p20 = out[:, :, :, 20]
        tem_v20 = out_fromv[:, :, :, 20]
        tem_p20 = tf.expand_dims(tem_p20, axis=3)
        tem_v20 = tf.expand_dims(tem_v20, axis=3)
        y_20 = tf.concat([tem_p20, tem_v20], axis=3)
        y_20 = tf.layers.conv2d(y_20, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat20')

        tem_p21 = out[:, :, :, 21]
        tem_v21 = out_fromv[:, :, :, 21]
        tem_p21 = tf.expand_dims(tem_p21, axis=3)
        tem_v21 = tf.expand_dims(tem_v21, axis=3)
        y_21 = tf.concat([tem_p21, tem_v21], axis=3)
        y_21 = tf.layers.conv2d(y_21, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat21')

        tem_p22 = out[:, :, :, 22]
        tem_v22 = out_fromv[:, :, :, 22]
        tem_p22 = tf.expand_dims(tem_p22, axis=3)
        tem_v22 = tf.expand_dims(tem_v22, axis=3)
        y_22 = tf.concat([tem_p22, tem_v22], axis=3)
        y_22 = tf.layers.conv2d(y_22, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat22')

        tem_p23 = out[:, :, :, 23]
        tem_v23 = out_fromv[:, :, :, 23]
        tem_p23 = tf.expand_dims(tem_p23, axis=3)
        tem_v23 = tf.expand_dims(tem_v23, axis=3)
        y_23 = tf.concat([tem_p23, tem_v23], axis=3)
        y_23 = tf.layers.conv2d(y_23, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat23')

        tem_p24 = out[:, :, :, 24]
        tem_v24 = out_fromv[:, :, :, 24]
        tem_p24 = tf.expand_dims(tem_p24, axis=3)
        tem_v24 = tf.expand_dims(tem_v24, axis=3)
        y_24 = tf.concat([tem_p24, tem_v24], axis=3)
        y_24 = tf.layers.conv2d(y_24, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat24')

        tem_p25 = out[:, :, :, 25]
        tem_v25 = out_fromv[:, :, :, 25]
        tem_p25 = tf.expand_dims(tem_p25, axis=3)
        tem_v25 = tf.expand_dims(tem_v25, axis=3)
        y_25 = tf.concat([tem_p25, tem_v25], axis=3)
        y_25 = tf.layers.conv2d(y_25, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat25')

        tem_p26 = out[:, :, :, 26]
        tem_v26 = out_fromv[:, :, :, 26]
        tem_p26 = tf.expand_dims(tem_p26, axis=3)
        tem_v26 = tf.expand_dims(tem_v26, axis=3)
        y_26 = tf.concat([tem_p26, tem_v26], axis=3)
        y_26 = tf.layers.conv2d(y_26, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat26')

        tem_p27 = out[:, :, :, 27]
        tem_v27 = out_fromv[:, :, :, 27]
        tem_p27 = tf.expand_dims(tem_p27, axis=3)
        tem_v27 = tf.expand_dims(tem_v27, axis=3)
        y_27 = tf.concat([tem_p27, tem_v27], axis=3)
        y_27 = tf.layers.conv2d(y_27, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat27')

        tem_p28 = out[:, :, :, 28]
        tem_v28 = out_fromv[:, :, :, 28]
        tem_p28 = tf.expand_dims(tem_p28, axis=3)
        tem_v28 = tf.expand_dims(tem_v28, axis=3)
        y_28 = tf.concat([tem_p28, tem_v28], axis=3)
        y_28 = tf.layers.conv2d(y_28, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat28')

        tem_p29 = out[:, :, :, 29]
        tem_v29 = out_fromv[:, :, :, 29]
        tem_p29 = tf.expand_dims(tem_p29, axis=3)
        tem_v29 = tf.expand_dims(tem_v29, axis=3)
        y_29 = tf.concat([tem_p29, tem_v29], axis=3)
        y_29 = tf.layers.conv2d(y_29, 1, 1, padding='same', activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='concat29')

        # for j in range(seq_length - input_length):
        #     tem = out[:, :, :, j]
        #     tem_v = out_fromv[:, :, :, j]
        #     tem = tf.expand_dims(tem, axis=3)
        #     tem_v = tf.expand_dims(tem_v, axis=3)
        #     y_0 = tf.concat([tem, tem_v], axis=3)
        #     if j == 0:
        #         z = y_0
        #     else:
        z = tf.concat([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9,
                       y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21,
                       y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29], axis=3)

        z = TB('TrajBlockout', filter_size, num_hidden, keep_prob)(z)
        z = tf.layers.conv2d(z, num_hidden[0], filter_size, padding='same', activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='2str_out_conv0')
        z = TB('TrajBlockout_1', filter_size, num_hidden, keep_prob)(z)
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
