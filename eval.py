"""
验证model准确率
"""
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from load import load_dataset
from cnn_model import cnn_model_fn


PATH = "data/test/"


def eval_model(model_path):
    """
    测试model准确率
    """

    eval_data, eval_labels = load_dataset(PATH)

    # 创建模型
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)

    # 评估模型和输出结果
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=eval_labels,
        num_epochs=1,
        shuffle=True
    )

    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)

    print("----------------------------------------\n\
    总共训练步数：{g_step}\n\
    测试图片数量: {num}\n\
    loss 值: {loss:0.4f}\n\
    识别准确率: {accuracy:0.2f}%\
    \n----------------------------------------\n".format(
        g_step=eval_results["global_step"],
        loss=eval_results["loss"],
        num=eval_data.shape[0],
        accuracy=eval_results["accuracy"] * 100
    ))


if __name__ == "__main__":
    eval_model("models/cnn_model")
    # eval_model("models/fm_cnn_model_x64")
