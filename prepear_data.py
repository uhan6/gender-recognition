"""
把数据集处理为合适的结构
取原数据集中 2 次测量1151张图片
分为921张训练
230张测试
处理后图片为 32 * 32 * 3
"""
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import, division, print_function

import os
import shutil
import random

import numpy as np
from PIL import Image

IMG_W = 32
IMG_H = 32
IMG_C = 3


def copy_dataset_by_filename(in_src, out_src):
    """
    数据分为训练集和测试集并存储到 out_src
    训练 921 张角度脸（男女）
    测试 230 张角度脸 93 男 137 女
    训练和测试数据不重复，数据内部有人物重复，图片不重复
    """

    # 得到所有文件
    all_file_names = os.listdir(in_src)

    # 筛选后的文件
    file_names = []

    # 文件名格式为CF0001_1101_00F.jpg
    # 判断 '2' 为 有缩放第二次测量
    for name in all_file_names:
        if not name[-12] == '2':
            continue

        file_names.append(name)

    # 随机获得 93个男和137个女的文件名集合
    male_name_set, female_name_set = get_random_man_feman_file_name(
        file_names, 93, 137)

    # 创建目录
    if not os.path.exists(out_src):
        os.mkdir(out_src)
        os.mkdir(out_src + "train/")
        os.mkdir(out_src + "test/")

    index_train = 0
    index_test = 0
    for name in file_names:
        if name in male_name_set or name in female_name_set:
            # 复制
            shutil.copyfile(in_src + name, out_src + "test/" +
                            name[1] + "_" + str(index_test) + ".jpg")
            index_test += 1
        else:
            shutil.copyfile(in_src + name, out_src + "train/" +
                            name[1] + "_" + str(index_train) + ".jpg")
            index_train += 1


# 弃用
# 能把图片转换为32 * 32 并存储，但是效率低
def load_dataset(in_src, out_src):
    """
    读取数据，分为训练集和测试集并存储
    训练 921 张角度脸（男女）
    测试 230 张角度脸 93 男 137 女
    训练和测试数据不重复，数据内部有人物重复，图片不重复
    """
    file_names = os.listdir(in_src)

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

        img = Image.open(in_src + file_name)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        _data = img.getdata()
        _data = np.array(_data, dtype=np.float32)

        # 将数据转为一维，不需要这一步,因为append()会按一维添加
        # _data = np.concatenate(_data)

        _label = 0
        if file_name[1] == 'M':
            _label = 1

        datasets["all_data"] = np.append(datasets["all_data"], _data)
        datasets["all_label"] = np.append(datasets["all_label"], _label)

    # 转换维度
    datasets["all_data"] = np.reshape(
        datasets["all_data"], (-1, IMG_W * IMG_H * IMG_C))

    # 这里获取的是文件的索引，效率比较低
    man_index_set, feman_index_set = get_random_man_feman_index(
        datasets["all_data"], datasets["all_label"], 93, 137)

    count_train_f = 0
    count_train_m = 0
    count_test_f = 0
    count_test_m = 0

    # 从 datasets 中取出随机出来的图片
    for i in range(0, datasets["all_data"].shape[0]):
        if i in man_index_set or i in feman_index_set:
            datasets["test_images"] = np.append(
                datasets["test_images"], datasets['all_data'][i])
            datasets["test_labels"] = np.append(
                datasets["test_labels"], datasets['all_label'][i])
            if datasets["all_label"][i] == 0:
                count_test_f += 1
            else:
                count_test_m += 1
        else:
            datasets["train_images"] = np.append(
                datasets["train_images"], datasets['all_data'][i])
            datasets["train_labels"] = np.append(
                datasets["train_labels"], datasets['all_label'][i])
            if datasets["all_label"][i] == 0:
                count_train_f += 1
            else:
                count_train_m += 1
    datasets["train_images"] = np.reshape(
        datasets["train_images"], (-1, IMG_W * IMG_H * IMG_C))
    datasets["test_images"] = np.reshape(
        datasets["test_images"], (-1, IMG_W * IMG_H * IMG_C))

    # 创建目录
    if not os.path.exists(out_src):
        os.mkdir(out_src)
        os.mkdir(out_src + "train/")
        os.mkdir(out_src + "test/")

    index = 0
    index_f = 0
    index_m = 0
    # 储存训练数据
    for data in datasets["train_images"]:
        data = np.asarray(data, dtype=np.uint8)
        data = data.reshape(IMG_W, IMG_H, IMG_C)
        img = Image.fromarray(data)

        if datasets["train_labels"][index] == 0:
            path = "%strain/F_%d.jpg" % (out_src, index_f)
            index_f += 1
        else:
            path = "%strain/M_%d.jpg" % (out_src, index_m)
            index_m += 1

        img.save(path)
        index += 1

    index = 0
    index_f = 0
    index_m = 0
    # 储存测试数据
    for data in datasets["test_images"]:
        data = np.asarray(data, dtype=np.uint8)
        data = data.reshape(IMG_W, IMG_H, IMG_C)
        img = Image.fromarray(data)

        if datasets["test_labels"][index] == 0:
            path = "%stest/F_%d.jpg" % (out_src, index_f)
            index_f += 1
        else:
            path = "%stest/M_%d.jpg" % (out_src, index_m)
            index_m += 1

        img.save(path)
        index += 1

    print("----------------------------------------\n\
    总图片数量: {all_num}\n\
    训练图片数量: {train_num}    男: {train_m} 女: {train_f}\n\
    验证图片数量: {test_num}    男: {test_m} 女: {test_f}\n\
    存储目录: {out_src}\
    \n----------------------------------------\n".format(
        all_num=datasets["all_data"].shape[0],
        train_num=datasets["train_images"].shape[0],
        test_num=datasets["test_images"].shape[0],
        train_m=count_train_m,
        train_f=count_train_f,
        test_m=count_test_m,
        test_f=count_test_f,
        out_src=out_src
    ))

    return datasets


def get_random_man_feman_file_name(file_names, male_num, female_num):
    """
    从file_names中随机选取一批男女照片
    返回 文件名 list
    不重复
    """

    male_file_name_set = set()
    female_file_name_set = set()

    while len(male_file_name_set) < male_num or len(female_file_name_set) < female_num:
        index = random.randint(0, len(file_names) - 1)

        if file_names[index][1] == 'M' and len(male_file_name_set) < male_num:
            male_file_name_set.add(file_names[index])
        elif file_names[index][1] == 'F' and len(female_file_name_set) < female_num:
            female_file_name_set.add(file_names[index])

    return male_file_name_set, female_file_name_set


def get_random_man_feman_index(datasets, labels, male_num, female_num):
    """
    从index中随机选取一批index
    不重复
    """
    male_index_set = set()
    female_index_set = set()

    while len(male_index_set) < male_num or len(female_index_set) < female_num:
        index = random.randint(0, datasets.shape[0] - 1)

        if labels[index] == 1 and len(male_index_set) < male_num:
            male_index_set.add(index)
        elif labels[index] == 0 and len(female_index_set) < female_num:
            female_index_set.add(index)

    return male_index_set, female_index_set


if __name__ == "__main__":
    copy_dataset_by_filename("Caucasian/", "data/")
    # load_dataset("Caucasian/", "data/")
