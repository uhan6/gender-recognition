"""
训练模型
"""
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from load import load_dataset
from cnn_model import cnn_model_fn


PATH = "data/train/"

tf.logging.set_verbosity(tf.logging.INFO)


def train_model(model_path):
    """
    测试model准确率
    """

    train_data, train_labels = load_dataset(PATH)

    # 删除旧模型
    # TODO 无法删除非空目录
    # os.rmdir("models/fm_cnn_model")

    # 注意：每次训练会叠加，除非删除原model目录
    # 创建估计模型
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)

    logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=100)

    # 训练模型
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    # 1600 步长在此问题已经接近最优
    # 1500 0.9087 loss 0.2184
    # 1600 0.9261 loss 0.2053
    # 1700 0.9174 loss 0.2160
    cifar10_classifier.train(
        input_fn=train_input_fn,
        steps=1600,
        hooks=[logging_hook]
    )


if __name__ == "__main__":
    train_model("models/cnn_model")
