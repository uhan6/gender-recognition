# 性别识别

[![license](https://img.shields.io/github/license/go88/gender-recognition.svg?style=for-the-badge)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/go88/gender-recognition/pulls)
[![GitHub (pre-)release](https://img.shields.io/github/release/go88/gender-recognition/all.svg?style=for-the-badge)](https://github.com/go88/gender-recognition/releases)

通过CNN神经网络，把输入的人脸图片分辨为男性或女性

## 数据集要求：

[CNBC数据集](http://wiki.cnbc.cmu.edu/Face_Place)

1. 下载后将Caucasian数据文件放入项目目录下
2. 运行prepear_data.py把数据分割，自动生成data目录存放分割后的数据
3. 运行train.py训练，eval.py测试。
