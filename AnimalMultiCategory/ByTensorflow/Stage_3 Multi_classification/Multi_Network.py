#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/25 20:56
# @Author : Micky
# @Desc : 多标签分类网络
# @File : Multi_Network.py
# @Software: PyCharm

import os
import Helper
import tensorflow as tf
import numpy as np


class Net(object):
    def __init__(self, width, height, channel, n_classify_c, n_classify_s, epochs, batch_size):
        """
        传入一些参数
        :param width: 传入原始图片宽度
        :param height: 传入原始图片高度
        :param channel: 传入原始图片通道数
        :param n_classify_c: 类别数量
        :param n_classify_s: 类别数量
        :param epochs: 迭代次数
        :param batch_size: 每次迭代的样本个数
        """
        self.width = width
        self.height = height
        self.channel = channel
        self.n_classify_c = n_classify_c
        self.n_classify_s = n_classify_s
        self.epochs = epochs
        self.batch_size = batch_size
        self.x = None
        self.y_classes = None
        self.y_species = None
        self.training = None
        self.keepdrop = None
        self.lr = None
        self.logits_c = None
        self.logits_s = None
        self.opt_train = None
        self.loss = None
        self.accuracy = None
        self.prob_c = None
        self.prob_s = None

    def _buildGraph(self):
        """
        构建图
        :return:
        """
        # 定义占位符
        input_x = tf.placeholder(tf.float32, shape=[None, self.width, self.height, self.channel], name='input_x')
        input_y_classes = tf.placeholder(tf.float32, shape=[None, self.n_classify_c], name='input_y_classes')
        input_y_species = tf.placeholder(tf.float32, shape=[None, self.n_classify_s], name='input_y_species')
        self.x = input_x
        self.y_classes = input_y_classes
        self.y_species = input_y_species
        self.training = tf.placeholder(tf.bool, name='training')
        self.keepdrop = tf.placeholder(tf.float32, name='keep_drop')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # 执行卷积过程
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(input_x, 64, kernel_size=7, strides=2, padding='same')
            conv1 = tf.layers.batch_normalization(conv1, training=self.training)
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding='same')
            conv1 = tf.nn.dropout(conv1, keep_prob=self.keepdrop)
        # 执行第一个残差块
        # [None, 56, 56, 64]  -----》 [None, 56, 56, 256]
        with tf.variable_scope('conv2'):
            # 做了resize，但strides仍然默认为1
            conv2 = self._resnet_bottleneck_block(conv1, std_filters=64, resize=True)
            conv2 = self._resnet_bottleneck_block(conv2, std_filters=64, resize=False)
            conv2 = self._resnet_bottleneck_block(conv2, std_filters=64, resize=False)
            conv2 = tf.nn.dropout(conv2, keep_prob=self.keepdrop)

        # [None, 28, 28, 64]  -----》 [None, 28, 28, 512]
        with tf.variable_scope('conv3'):
            # 高宽减半了。
            conv3 = self._resnet_bottleneck_block(conv2, std_filters=128, resize=True, block_stride=2)
            conv3 = self._resnet_bottleneck_block(conv3, std_filters=128, resize=False)
            conv3 = tf.nn.dropout(conv3, keep_prob=self.keepdrop)
            conv3 = self._resnet_bottleneck_block(conv3, std_filters=128, resize=False)
            conv3 = self._resnet_bottleneck_block(conv3, std_filters=128, resize=False)
            conv3 = tf.nn.dropout(conv3, keep_prob=self.keepdrop)

        # [None, 28, 28, 64]  -----》 [None, 14, 14, 1024]
        with tf.variable_scope('conv4'):
            # 高宽减半了。
            conv4 = self._resnet_bottleneck_block(conv3, std_filters=256, resize=True, block_stride=2)
            conv4 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=False)
            conv4 = tf.nn.dropout(conv4, keep_prob=self.keepdrop)
            conv4 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=False)
            conv4 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=False)
            conv4 = tf.nn.dropout(conv4, keep_prob=self.keepdrop)
            conv4 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=False)
            conv4 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=False)
            conv4 = tf.nn.dropout(conv4, keep_prob=self.keepdrop)

        # [None, 14, 14, 1024]  -----》 [None, 7, 7, 2048]
        # [None, 14, 14, 1024]  -----》 [None, 7, 7, 1024]  这里是因为电脑要求，最后一层卷积操我没有用2048的深度
        with tf.variable_scope('conv5'):
            # 高宽减半了。
            conv5 = self._resnet_bottleneck_block(conv4, std_filters=256, resize=True, block_stride=2)
            conv5 = self._resnet_bottleneck_block(conv5, std_filters=256, resize=False)
            conv5 = self._resnet_bottleneck_block(conv5, std_filters=256, resize=False)
            conv5 = tf.nn.dropout(conv5, keep_prob=self.keepdrop)

        # 执行全局平均池化
        pool1 = tf.layers.average_pooling2d(conv5, pool_size=7, strides=1)
        pool1 = tf.nn.dropout(pool1, keep_prob=self.keepdrop)

        # 拉平层
        flatten1 = tf.layers.flatten(pool1, name='flatten')
        # pool1 = tf.reshape(pool1, [-1, 2048])

        # # 全连接层
        with tf.variable_scope('fc1'):
            # 全连接层
            fc1 = tf.layers.dense(flatten1, 1000, activation=tf.nn.relu)
            fc1 = tf.nn.dropout(fc1, keep_prob=self.keepdrop)
        # 输出层
        with tf.variable_scope('out'):
            logits_c = tf.layers.dense(fc1, self.n_classify_c)
            prob_c = tf.nn.softmax(logits_c)

            logits_s = tf.layers.dense(fc1, self.n_classify_s)
            prob_s = tf.nn.softmax(logits_s)

        # 计算损失  两个损失函数相加
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_c, labels=input_y_classes)) \
            + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_s, labels=input_y_species))

        # 定义优化器
        optimizer = tf.train.AdamOptimizer(self.lr)
        # 代码种有BN操作的话，需要这么操作
        # 这里是为了更新BN中的两个参数后在去最小化损失
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_train = optimizer.minimize(loss)

        # 计算准确率
        correct_pred_c = tf.equal(tf.argmax(logits_c, axis=1), tf.argmax(input_y_classes, axis=1))
        correct_pred_s = tf.equal(tf.argmax(logits_s, axis=1), tf.argmax(input_y_species, axis=1))

        # boolean值转float并求平均得到准确率
        accuracy = (tf.reduce_mean(tf.cast(correct_pred_c, tf.float32)) + tf.reduce_mean(tf.cast(correct_pred_s, tf.float32))) / 2

        self.logits_c = logits_c
        self.logits_s = logits_s
        self.opt_train = opt_train
        self.loss = loss
        self.accuracy = accuracy
        self.prob_c = prob_c
        self.prob_s = prob_s


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

    def train(self, X_train, y_train_classes, y_train_species, X_val, y_val_classes, y_val_species):
        """
        训练模型
        :param X_train: 训练集图片样本
        :param y_train_classes: 训练集图片样本对应的label
        :param y_train_species: 训练集图片样本对应的label
        :param X_val: 验证集图片样本
        :param y_val_classes: 验证集图片样本
        :param y_val_species: 验证集图片样本
        :return:
        """
        # 构建图
        self._buildGraph()

        # 训练模型保存
        saver = tf.train.Saver()
        CHECKPOINT_DIR = './models'
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        checkpoint = './models/best_model.ckpt'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 模型恢复
            if os.path.exists(checkpoint+'.meta'):
                saver.restore(sess, checkpoint)
            for epoch in range(self.epochs):
                train_acc_sum = []
                train_loss_sum = []
                for features, labels_c, labels_s in Helper.batch_features_labels2(X_train, y_train_classes, y_train_species, self.batch_size):
                    feed_dict = {
                        self.x: features,
                        self.y_classes: labels_c,
                        self.y_species: labels_s,
                        self.keepdrop: 0.5,
                        self.lr: 0.001,
                        self.training: True
                    }
                    _, train_loss, train_accuracy = sess.run([self.opt_train, self.loss, self.accuracy], feed_dict=feed_dict)
                    train_acc_sum.append(train_accuracy)
                    train_loss_sum.append(train_loss)

                saver.save(sess, checkpoint)
                print('当前的epoch:{}, 训练集损失是:{}, 训练集准确率是:{}'.format(
                    epoch, np.mean(train_loss_sum), np.mean(train_acc_sum)
                ))


                # 验证集
                valid_acc_sum = []
                valid_loss_sum = []
                for features, labels_c, labels_s in Helper.batch_features_labels2(X_val, y_val_classes, y_val_species, self.batch_size):
                    feed_dict = {
                        self.x: features,
                        self.y_classes: labels_c,
                        self.y_species: labels_s,
                        self.keepdrop: 1.0,
                        self.lr: 0.001,
                        self.training: False
                    }
                    val_loss, val_accuracy = sess.run([self.loss, self.accuracy],
                                                             feed_dict=feed_dict)
                    valid_acc_sum.append(val_accuracy)
                    valid_loss_sum.append(val_loss)
                print('当前的epoch:{}, 验证集损失是:{}, 验证集准确率是:{}'.format(
                    epoch, np.mean(valid_loss_sum), np.mean(valid_acc_sum)
                ))

                # 写入日志
                content = '当前的epoch: {}, 训练集损失: {}, 训练集准确率: {}, 验证集损失: {}, 验证集准确率: {}'.format(epoch, np.mean(train_loss_sum), np.mean(train_acc_sum), np.mean(valid_loss_sum), np.mean(valid_acc_sum))
                Helper.wirteTrainLog('多标签分类模型训练日志', content)
                if np.mean(valid_acc_sum) >= 0.88:
                    break

    def predict(self, X):
        """
        分类预测
        :param X:  预测样本
        :return:
        """
        prob_c, prob_s = self.probability(X)
        return np.argmax(prob_c, axis=1), np.argmax(prob_s, axis=1)


    def probability(self, X):
        """
        预测概率
        :param X: 预测样本
        :return: 各类别概率
        """
        X = np.asarray(X)
        checkpoint = './models/best_model.ckpt'
        # 模型恢复
        if not os.path.exists(checkpoint + '.meta'):
            raise FileExistsError('未找到训练好的模型！！！ 请先运行Classes_classification.py进行模型运行！！！')

        with tf.Graph().as_default():
            self._buildGraph()
            save = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                save.restore(sess, checkpoint)
                feed_dict = {
                    self.x: X,
                    self.keepdrop: 1,
                    self.lr: 0.001,
                    self.training: False
                }
                prob_c, prob_s = sess.run([self.prob_c, self.prob_s], feed_dict=feed_dict)
                return prob_c, prob_s