"""
读取数据
"""
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
from PIL import Image


def load_dataset(src):
    """
    读取训练数据或测试数据

    参数精确到类别目录: data/train/

    训练 921 张角度脸（男女）用于训练 374 男 547 女
    测试 230 张角度脸（男女）用于测试  93 男 137 女

    训练和测试数据不重复，数据内部有人物重复，图片不重复
    """

    # 获取 src 下的所有文件名
    file_names = os.listdir(src)

    # 使用 array 初始化 要传参数 []
    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int32)

    # f female 女
    # m male 男
    count_f = 0
    count_m = 0
    for file_name in file_names:
        img = Image.open(src + file_name)
        # 默认读取是 gray，需要转换
        img = img.convert("RGB")
        # 把图片转换为合适大小
        img = img.resize((32, 32))
        # 获取图片的数据，转为 np.float32，用 int32 也可以，tensorflow 默认数据为 float
        _data = img.getdata()
        _data = np.array(_data, dtype=np.float32)

        # 标签 0 女 1 男
        _label = 0
        if file_name[0] == 'M':
            _label = 1
            count_m += 1
        else:
            count_f += 1

        # 每张图片和标签都顺序加在后面
        images = np.append(images, _data)
        labels = np.append(labels, _label)

    # 把数据分为二维数组，每一行代表一张图片的全部内容 32 * 32 * 3
    images = np.reshape(images, (-1, 3072))

    print("----------------------------------------\n\
    读取图片数量: {all_num}\n\
    男: {m} 女: {f}\
    \n----------------------------------------\n".format(
        all_num=images.shape[0],
        m=count_m,
        f=count_f
    ))

    return images, labels


# 弃用
# 刚开始的测试，直接从 Caucasian 里拿数据
def load_dataset_simple(src):
    """
    读取整合数据
    训练 40 * 2 张正脸（男女）
    测试 10 * 2 张正脸（男女）
    训练和测试数据不重复，数据内部不重复
    """

    file_names = os.listdir(src)

    # 用 dict 来存每一个数据，初始化同上
    datasets = {"train_images": np.array([], dtype=np.float32),
                "train_labels": np.array([], dtype=np.int32),
                "test_images": np.array([], dtype=np.float32),
                "test_labels": np.array([], dtype=np.int32)}

    count_f = 0
    count_m = 0
    # 文件名格式为CF0001_1101_00F.jpg
    # 判断 '1' 为 无缩放第一次测量 'F' 为正脸
    for file_name in file_names:
        if not (file_name[-12] == '1' and file_name[-5] == 'F'):
            continue

        # 同上
        img = Image.open(src + file_name)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        _data = img.getdata()
        _data = np.array(_data, dtype=np.float32)

        _label = 0
        if file_name[1] == 'M':
            _label = 1

        # 总共 58 女 51 男
        # 40女40男 训练 10女10男 测试
        if _label == 0:
            if count_f < 40:
                datasets["train_images"].append(_data)
                datasets["train_labels"].append(_label)
            elif count_f < 50:
                datasets["test_images"].append(_data)
                datasets["test_labels"].append(_label)
            count_f += 1
        else:
            if count_m < 40:
                datasets["train_images"].append(_data)
                datasets["train_labels"].append(_label)
            elif count_m < 50:
                datasets["test_images"].append(_data)
                datasets["test_labels"].append(_label)
            count_m += 1

    # 切分同上
    datasets["train_images"] = np.reshape(datasets["train_images"], (-1, 3072))
    datasets["test_images"] = np.reshape(datasets["test_images"], (-1, 3072))

    print("--------------------\n训练图片数量: %d  验证图片数量: %d \n--------------------\n"
          % (datasets["train_images"].shape[0], datasets["test_images"].shape[0]))

    return datasets


# 弃用
# 从 Caucasian 里拿大部分数据
def load_dataset_complex(src):
    """
    读取整合数据
    训练 920 张角度脸（男女）
    测试 231 张角度脸（男女）
    训练和测试数据不重复，数据内部有重复
    """

    file_names = os.listdir(src)

    datasets = {"all_data": np.array([], dtype=np.float32),
                "all_label": np.array([], dtype=np.int32),
                "train_images": np.array([], dtype=np.float32),
                "train_labels": np.array([], dtype=np.int32),
                "test_images": np.array([], dtype=np.float32),
                "test_labels": np.array([], dtype=np.int32)}

    # 文件名格式为CF0001_1101_00F.jpg
    # 判断 '2' 为 有缩放第二次测量
    for file_name in file_names:
        if not file_name[-12] == '2':
            continue

        img = Image.open(src + file_name)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        _data = img.getdata()
        _data = np.array(_data, dtype=np.float32)

        _data = np.concatenate(_data)

        _label = 0
        if file_name[1] == 'M':
            _label = 1

        datasets["all_data"] = np.append(datasets["all_data"], _data)
        datasets["all_label"] = np.append(datasets["all_label"], _label)

    # 把数据分为二维数组，每一行代表一张图片的全部内容 32 * 32 * 3
    datasets["all_data"] = np.reshape(datasets["all_data"], (-1, 3072))

    # 随机选取 231 张图片用于测试，剩下的用于训练
    # 返回的是一个索引的 list
    random_index = get_random_index(datasets["all_data"].shape[0], 231)

    count_train_f = 0
    count_train_m = 0
    count_test_f = 0
    count_test_m = 0

    # 遍历数据集，如果不是是测试集的索引，就放入训练集，反之放入测试集
    for i in range(0, datasets['all_data'].shape[0]):
        if i not in random_index:
            datasets["train_images"] = np.append(
                datasets["train_images"], datasets['all_data'][i])
            datasets["train_labels"] = np.append(
                datasets["train_labels"], datasets['all_label'][i])

            if datasets['all_label'][i] == 0:
                count_train_f += 1
            else:
                count_train_m += 1
        else:
            datasets["test_images"] = np.append(
                datasets["test_images"], datasets['all_data'][i])
            datasets["test_labels"] = np.append(
                datasets["test_labels"], datasets['all_label'][i])

            if datasets['all_label'][i] == 0:
                count_test_f += 1
            else:
                count_test_m += 1
    datasets["train_images"] = np.reshape(datasets["train_images"], (-1, 3072))
    datasets["test_images"] = np.reshape(datasets["test_images"], (-1, 3072))

    print("----------------------------------------\n\
    总图片数量: {all_num}\n\
    训练图片数量: {train_num}    男: {train_m} 女: {train_f}\n\
    验证图片数量: {test_num}    男: {test_m} 女: {test_f}\
    \n----------------------------------------\n".format(
        all_num=datasets["all_data"].shape[0],
        train_num=datasets["train_images"].shape[0],
        test_num=datasets["test_images"].shape[0],
        train_m=count_train_m,
        train_f=count_train_f,
        test_m=count_test_m,
        test_f=count_test_f
    ))

    return datasets


def get_random_index(all_index, batch):
    """
    从index中随机选取一批index
    不重复
    """

    # 使用集合，内部元素不重复
    index_set = set()

    while len(index_set) < batch:
        index = random.randint(0, all_index)

        index_set.add(index)

    return index_set


if __name__ == "__main__":
    load_dataset("data/train/")
