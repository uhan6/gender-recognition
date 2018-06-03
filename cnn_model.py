"""
定义卷积神经网络的模型函数
"""
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """CNN的模型函数"""
    # 输入层
    # 将数据分为4维张量[batch_size, width, height, channels]
    # 图片 32x32 有3个颜色通道

    input_layer = tf.reshape(features, (-1, 32, 32, 3))

    # 卷积层 #1
    # 用relu激活的 5x5 的核心计算 32 个特征
    # 输入层张量：[batch_size, 32, 32, 3]
    # 输出层张量：[batch_size, 32, 32, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # 池化层 #1
    # 第一个使用最大池化 池大小 3x3 步长 2
    # 输入层张量：[batch_size, 32, 32, 32]
    # 输出层张量：[batch_size, 15, 15, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=2
    )

    # 卷积层 #2
    # 用relu激活的 5x5 的核心计算 32 个特征
    # 输入层张量：[batch_size, 15, 15, 32]
    # 输出层张量：[batch_size, 15, 15, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # 池化层 #2
    # 平均池化 池大小 3x3 步长 2
    # 输入层张量：[batch_size, 15, 15, 32]
    # 输出层张量：[batch_size, 7, 7, 32]
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=2
    )

    # 卷积层 #3
    # 用relu激活的 5x5 的核心计算 64 个特征
    # 初始权值0.01
    # 输入层张量：[batch_size, 7, 7, 32]
    # 输出层张量：[batch_size, 7, 7, 64]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # 池化层 #3
    # 平均池化 池大小 3x3 步长 2
    # 输入层张量：[batch_size, 7, 7, 64]
    # 输出层张量：[batch_size, 3, 3, 64]
    pool3 = tf.layers.average_pooling2d(
        inputs=conv3,
        pool_size=[3, 3],
        strides=2
    )

    # 将张量变成向量
    # 输入层张量: [batch_size, 3, 3, 64]
    # 输出层张量: [batch_size, 3 * 3 * 64]
    pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 64])

    # 密集层
    # 1024个神经元连接的密集层
    # 输入层张量: [batch_size, 3 * 3 * 64]
    # 输出层张量: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    # 加上丢弃操作; 0.6概率数据会保存
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits 层
    # 输入层张量: [batch_size, 1024]
    # 输出层张量: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # 生成预测 (PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # 添加"softmax_tensor"到图层中. 被"logging_hook"用于 预测(PREDICT)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 计算损失 (TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 设置训练操作 (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 添加评估指标 (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
