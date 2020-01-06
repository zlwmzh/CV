#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/29 21:42
# @Author : Micky
# @Desc : 鼻子的关键点检测网络
# @File : model_nose_net.py
# @Software: PyCharm

import tensorflow as tf
import ConfigN
import os


current_path = os.path.dirname(__file__)
print('当前路径：{}'.format(current_path))
class NNet:
    def __init__(self, width, height, channel, n_classify, epochs, batch_size):
        """
        传入一些参数
        :param width: 传入原始图片宽度
        :param height: 传入原始图片高度
        :param channel: 传入原始图片通道数
        :param n_classify: 回归的最终个数。 这里是4*4 = 16
        :param epochs: 迭代次数
        :param batch_size: 每次迭代的样本个数
        """
        self.width = width
        self.height = height
        self.channel = channel
        self.n_classify = n_classify
        self.epochs = epochs
        self.batch_size = batch_size
        self.x = None
        self.y = None
        self.training = None
        self.keepdrop = None
        self.lr = None
        self.logits = None
        self.opt_train = None
        self.loss = None
        self.accuracy = None
        self.prob = None

        # self.checkpoint = './model/best_model.ckpt'
        # CHECKPOINT_DIR = './models'
        # if not os.path.exists(CHECKPOINT_DIR):
        #     os.makedirs(CHECKPOINT_DIR)
        tf.reset_default_graph()
        self._buildGraph()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())
        # # 模型恢复
        # if os.path.exists(self.checkpoint + '.meta'):
        #     self.saver.restore(self.sess, self.checkpoint)

    def _buildGraph(self):
        """
        构建图
        :return:
        """
        # 定义占位符
        input_x = tf.placeholder(tf.float32, shape=[None, self.width, self.height, self.channel], name='input_x')
        input_y = tf.placeholder(tf.float32, shape=[None, self.n_classify], name='input_y')
        self.x = input_x
        self.y = input_y
        self.training = tf.placeholder(tf.bool, name='training')
        self.keepdrop = tf.placeholder(tf.float32, name='keep_drop')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # 执行卷积过程
        with tf.variable_scope('conv1_n'):
            conv1 = tf.layers.conv2d(input_x, 20, kernel_size=4, strides=1, padding='valid')
            # conv1 = tf.layers.batch_normalization(conv1, training=self.training)
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same')
            # conv1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same')
            # conv1 = tf.nn.dropout(conv1, keep_prob=self.keepdrop)
        # 执行第一个残差块
        # [None, 56, 56, 64]  -----》 [None, 56, 56, 256]
        with tf.variable_scope('conv2_n'):
            # # 做了resize，但strides仍然默认为1
            conv2 = tf.layers.conv2d(conv1, 40, kernel_size=3, strides=1, padding='valid')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same')
            # conv2 = self._resnet_bottleneck_block(conv1, std_filters=64, resize=True)
            # conv2 = self._resnet_bottleneck_block(conv2, std_filters=64, resize=False)
            # conv2 = self._resnet_bottleneck_block(conv2, std_filters=64, resize=False)
            # conv2 = tf.nn.dropout(conv2, keep_prob=self.keepdrop)

        # [None, 28, 28, 64]  -----》 [None, 28, 28, 512]
        with tf.variable_scope('conv3_n'):
            # 高宽减半了。
            conv3 = tf.layers.conv2d(conv2, 60, kernel_size=3, strides=1, padding='valid')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='same')

        # [None, 28, 28, 64]  -----》 [None, 14, 14, 1024]
        with tf.variable_scope('conv4_n'):
            conv4 = tf.layers.conv2d(conv3, 80, kernel_size=2, strides=1, padding='valid')
            conv4 = tf.nn.relu(conv3)
            # conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='same')

        # [None, 14, 14, 1024]  -----》 [None, 7, 7, 2048]
        # [None, 14, 14, 1024]  -----》 [None, 7, 7, 1024]  这里是因为电脑要求，最后一层卷积操我没有用2048的深度
        # with tf.variable_scope('conv5'):
        #     # 高宽减半了。
        #     conv5 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=True, block_stride=2)
        #     conv5 = self._resnet_bottleneck_block(conv5, std_filters=256, resize=False)
        #     conv5 = self._resnet_bottleneck_block(conv5, std_filters=256, resize=False)
        #     conv5 = tf.nn.dropout(conv5, keep_prob=self.keepdrop)

        # 执行全局平均池化
        # pool1 = tf.layers.average_pooling2d(conv5, pool_size=7, strides=1)
        # pool1 = tf.nn.dropout(pool1, keep_prob=self.keepdrop)

        # 拉平层
        flatten1 = tf.layers.flatten(conv4, name='flatten')
        # pool1 = tf.reshape(pool1, [-1, 2048])

        # # 全连接层
        with tf.variable_scope('fc1_n'):
            # 全连接层
            fc1 = tf.layers.dense(flatten1, 120, activation=tf.nn.relu)
            fc1 = tf.nn.dropout(fc1, keep_prob=self.keepdrop)
        # 输出层
        with tf.variable_scope('out_n'):
            logits = tf.layers.dense(fc1, self.n_classify)

            # prob = tf.nn.softmax(logits)

        # 计算损失 欧几里得距离
        loss = tf.losses.mean_squared_error(logits, input_y)
        # loss1 = tf.sqrt(tf.reduce_sum(tf.square(logits - input_y)))
        # hubers = tf.losses.huber_loss(logits, input_y)
        # loss = 0.8 * loss1 + 2 * loss2
        # 定义优化器
        optimizer = tf.train.AdamOptimizer(self.lr)
        # 代码种有BN操作的话，需要这么操作
        # 这里是为了更新BN中的两个参数后在去最小化损失
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_train = optimizer.minimize(loss)

        # 计算准确率
        # correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(input_y, axis=1))
        # boolean值转float并求平均得到准确率
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.logits = logits
        self.opt_train = opt_train
        self.loss = loss
        # self.accuracy = accuracy
        # self.prob = prob

        # 添加标量统计结果
        tf.summary.scalar("points_loss", loss)
        self.summary_op = tf.summary.merge_all()

    def _resnet_bottleneck_block(self, x, std_filters, resize=False, block_stride=1):
        """
        resnet残差块
        :param x: 输入数据
        :param std_filters: 卷积核个数
        :param resize: 是否对图片进行宽高减半的操作
        :param block_stride: 步长
        :return:
        """
        # 1. 实现图片的宽高进行减半的操作
        if resize:
            block_conv_input = tf.layers.conv2d(x, std_filters, kernel_size=1, strides=block_stride, padding='SAME')
            block_conv_input = tf.layers.batch_normalization(block_conv_input, training=self.training)
        else:
            # 就是 shortcut
            block_conv_input = x
        # 2. 实现残差块
        # 执行 1*1 卷积。 该卷积块 需要设置 block_stride 和 resize==True 情况下一样。
        y = tf.layers.conv2d(x, std_filters, kernel_size=1, strides=block_stride, padding='SAME')
        y = tf.layers.batch_normalization(y, training=self.training)
        y = tf.nn.relu(y)

        # 执行3*3 卷积
        y = tf.layers.conv2d(y, std_filters, kernel_size=3, padding='SAME')
        y = tf.layers.batch_normalization(y, training=self.training)
        y = tf.nn.relu(y)

        # 执行1*1卷积
        y = tf.layers.conv2d(y, std_filters * 4, kernel_size=1, padding='SAME')
        y = tf.layers.batch_normalization(y, training=self.training)
        y = tf.nn.relu(y)

        # 因为最后需要将y和block_conv_input 相加，所以需要判断channel是否等于std_filters*4
        if block_conv_input.shape[-1] != std_filters * 4:
            # 对shortcut再次做1*1卷积。 目的是为了 shortcut深度*4
            block_conv_input = tf.layers.conv2d(block_conv_input,
                                                std_filters * 4,
                                                kernel_size=1,
                                                padding='SAME')

        block_res = tf.add(y, block_conv_input)
        relu = tf.nn.relu(block_res)
        return relu

    def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        '''
        bbox_pred   ：预测框
        bbox_targets：标签框
        bbox_inside_weights：
        bbox_outside_weights：
        '''
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # tf.less 返回 True or False； a<b,返回True， 否则返回False。
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        # 实现公式中的条件分支
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (
        0.5 / sigma_2)) * (
                                                                                    1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    def train(self, model_save, base_dir, display):
        """
        模型训练
        :param model_save: 模型存放地址
        :param base_dir: 数据存放文件夹
        :param display: 多少条显示loss
        :return:
        """
        file_path = os.path.join(base_dir, 'tfrecord/nose.tfrecord_shuffle')
        # 读取tfrecord文件
        image_batch, label_batch = self.read_single_tfrecord(file_path)
        # 模型图存储文件夹
        logs_dir = '../graph/NNet/'
        # 如果不存在则创建文件夹
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        # 保存模型图
        writer = tf.summary.FileWriter(logs_dir, self.sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        i = 0

        label_file = os.path.join(base_dir, 'nose.txt')
        f_l = open(label_file, 'r')
        num = len(f_l.readlines())
        f_l.close()

        # 总共需要迭代多少步
        MAX_STEP = int(num / ConfigN.BOX_BATCH_SIZE + 1) * self.epochs
        epoch = 0
        self.sess.graph.finalize()

        try:
            for step in range(MAX_STEP):
                i = i + 1
                if coord.should_stop():
                    break
                image_batch_array, label_batch_array = self.sess.run(
                    [image_batch, label_batch])
                _, summary = self.sess.run([self.opt_train, self.summary_op],
                                           feed_dict={self.x: image_batch_array,
                                                      self.y: label_batch_array,
                                                      self.keepdrop: 0.5,
                                                      self.lr: 0.0001,
                                                      self.training: True})
                # 展示训练过程
                if (step + 1) % display == 0:
                    loss, lr = self.sess.run(
                        [self.loss, self.lr],
                        feed_dict={self.x: image_batch_array,
                                   self.y: label_batch_array,
                                   self.keepdrop: 0.5,
                                   self.lr: 0.0001,
                                   self.training: True
                                   })

                    print('epoch:%d/%d' % (epoch + 1, self.epochs))
                    print(
                        "Step: %d/%d, Loss: %4f ,lr:%f " % (
                            step + 1, MAX_STEP, loss, lr))

                # 每一次epoch 保留一次模型
                if i * ConfigN.BOX_BATCH_SIZE > num:
                    epoch += 1
                    i = 0
                    # 保存模型
                    self.saver.save(self.sess, model_save, global_step=epoch)
                writer.add_summary(summary, global_step=step)
        except tf.errors.OutOfRangeError:
            print("完成！！！")
        finally:
            coord.request_stop()
            writer.close()
        coord.join(threads)
        self.sess.close()

    def read_single_tfrecord(self, tfrecord_file):
        """
        读取tfrecord数据
        :param tfrecord_file: 文件路径
        :return:
        """
        filename_queue = tf.train.string_input_producer([tfrecord_file], shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        image_features = tf.parse_single_example(serialized_example,
                                                 features={
                                                     'image/encoded': tf.FixedLenFeature([], tf.string),
                                                     'image/label': tf.FixedLenFeature([8], tf.float32)
                                                 })
        image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
        image = tf.reshape(image, [ConfigN.INPUT_SIZE_BOX, ConfigN.INPUT_SIZE_BOX, 3])
        # 将值规划在[-1,1]内  0-255
        image = (tf.cast(image, tf.float32) - 127.5) / 128

        label = tf.cast(image_features['image/label'], tf.float32)
        image, label = tf.train.batch([image, label], batch_size=ConfigN.BOX_BATCH_SIZE,
                                              num_threads=2,
                                              capacity=ConfigN.BOX_BATCH_SIZE)
        label = tf.reshape(label, [ConfigN.BOX_BATCH_SIZE, 8])
        return image, label

    def predict(self, X, model_save_path):
        """
        分类预测
        :param X:  预测样本
        :param model_save_path: 模型路径
        :return:
        """
        model_path = model_save_path
        # 加载模型文件
        model_file = tf.train.latest_checkpoint(model_path)
        self.saver.restore(self.sess, model_file)

        feed_dict = {
            self.x: X,
            self.keepdrop: 1,
            self.lr: 0.0001,
            self.training: False
        }
        prob = self.sess.run([self.logits], feed_dict=feed_dict)
        return prob[0]